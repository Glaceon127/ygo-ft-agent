// 模型玩家插件：WindBot 作为占位客户端，插件接管它的对局响应
//
// 环境变量：
//   MODEL_PLAYER_NAME=model     占位客户端名字
//   MODEL_PLAYER_TAKEOVER=1     开启接管
//   MODEL_PLAYER_URL=http://127.0.0.1:3000
//   MODEL_PLAYER_POLL=5         routeGameMsg 轮询间隔(ms)
//   MODEL_PLAYER_TIMEOUT=400    routeGameMsg 轮询超时(ms)
/* eslint-disable @typescript-eslint/no-explicit-any */
import { BasePlugin, DefinePlugin } from '../../src/utility/base-plugin';
import { OnRoomDuelStart, OnRoomLeavePlayer, OnRoomWin, RoomManager } from '../../src/room';
import {
  IndexResponse,
  YGOProCtosResponse,
  YGOProMsgBase,
  YGOProMsgResponseBase,
  YGOProMsgRetry,
} from 'ygopro-msg-encode';
import { buildResponse } from './responses';
import { buildFieldSnapshot } from './input-builder';
import { ModelClient } from './model-client';
import { DuelObservation } from './duel-observation';
import * as fs from 'fs';
import * as path from 'path';
import { randomInt } from 'node:crypto';

const NAME = process.env.MODEL_PLAYER_NAME ?? 'model';
const TAKEOVER = process.env.MODEL_PLAYER_TAKEOVER === '1';
const POLL_INTERVAL = parseInt(process.env.MODEL_PLAYER_POLL ?? '5', 10);
const POLL_TIMEOUT = parseInt(process.env.MODEL_PLAYER_TIMEOUT ?? '400', 10);
const MODEL_URL = process.env.MODEL_PLAYER_URL ?? '';

// response(int) -> CTOS_RESPONSE 字节
function responseToBytes(msg: any, response: number): Uint8Array {
  const n = msg.constructor?.name ?? '';
  // select_place / select_disfield：response 是 places 的索引，必须回到位置编码
  if (/SelectPlace|SelectDisField/.test(n)) {
    const places = msg.getSelectablePlaces?.() ?? [];
    const p = places[response >= 0 ? response : 0] ?? places[0];
    if (p && msg.prepareResponse) {
      return msg.prepareResponse([
        { player: p.player, location: p.location, sequence: p.sequence },
      ]);
    }
  }
  // SelectUnselectCard: prepareResponse(cardOption?) 单参数
  //   -1 → 无参调用（finish/取消）；其他 → [1][i]
  if (/SelectUnselectCard/.test(n)) {
    if (msg.prepareResponse) {
      return response === -1
        ? msg.prepareResponse()
        : msg.prepareResponse(IndexResponse(response));
    }
  }
  // yesno / effectyn / position 等整数型响应必须写 4 字节 int32
  // 依据 ygoenv ygopro.h::YGO_SetResponsei：ReplayWriteInt8(4); ReplayWriteInt32(value);
  const buf = new Uint8Array(4);
  new DataView(buf.buffer).setInt32(0, response, true);
  return buf;
}

@DefinePlugin('model-player')
export class ModelPlayerPlugin extends BasePlugin {
  private logger = this.ctx.createLogger('ModelPlayer');
  private injected = 0;
  private swallowed = 0;
  private kinds = new Map<string, number>();
  private byHow = new Map<string, number>();
  private byHowBase?: Map<string, number>;
  private retries = 0;
  private stale = 0;
  private snapshots = 0;
  private oppIdle = 0;
  private injectedMsgs = new WeakSet<object>();
  private duelT0?: number;
  // 连锁栈状态（对齐 ygoenv feat(22..30)）
  private observation = new DuelObservation();
  private chainLogged = 0;
  // 埋点计数：DUEL_END 输出，缺一项就无法定位「模型没接管」
  private predFail = 0;
  private predSlow = 0;
  private predEmpty = 0;
  private annByChain = 0;
  private annNoChain = 0;
  private annModel = 0;
  private neg1Stop = 0;
  private sumUsed = 0;
  private chainAct = 0;
  private lastInjected = '';
  private model = new ModelClient(MODEL_URL, this.logger as any);

  // 预测耗时/失败埋点：慢/空/异常都是「连接/推理层」信号，必须与「某张卡的效果」区分开
  private async predictTimed(kind: string, input: any): Promise<any> {
    const t0 = Date.now();
    try {
      const pred = await this.model.predict(input);
      const ms = Date.now() - t0;
      if (!pred) {
        this.predEmpty += 1;
        this.logger.warn({ kind, ms, empty: this.predEmpty }, 'PREDICT_EMPTY');
      } else if (ms > 200) {
        this.predSlow += 1;
        this.logger.warn({ kind, ms, slow: this.predSlow }, 'PREDICT_SLOW');
      }
      return pred;
    } catch (e) {
      this.predFail += 1;
      this.logger.warn(
        { kind, ms: Date.now() - t0, error: e instanceof Error ? e.message : String(e), fail: this.predFail },
        'PREDICT_FAIL',
      );
      return null;
    }
  }

  /** 同 predictTimed，但返回按 prob 降序的完整候选表（多选场景用） */
  private async predictRankedTimed(kind: string, input: any): Promise<any[] | null> {
    const t0 = Date.now();
    try {
      const ranked = await this.model.predictRanked(input);
      const ms = Date.now() - t0;
      if (!ranked || !ranked.length) {
        this.predEmpty += 1;
        this.logger.warn({ kind, ms, empty: this.predEmpty }, 'PREDICT_EMPTY');
      } else if (ms > 200) {
        this.predSlow += 1;
        this.logger.warn({ kind, ms, slow: this.predSlow }, 'PREDICT_SLOW');
      }
      return ranked ?? null;
    } catch (e) {
      this.predFail += 1;
      this.logger.warn(
        { kind, ms: Date.now() - t0, error: e instanceof Error ? e.message : String(e), fail: this.predFail },
        'PREDICT_FAIL',
      );
      return null;
    }
  }

  async init(): Promise<void> {
    this.logger.info(
      { name: NAME, takeover: TAKEOVER, poll: POLL_INTERVAL, timeout: POLL_TIMEOUT },
      'ModelPlayer plugin loaded',
    );

    // 1) 吞掉占位客户端自己的对局响应（放行插件注入的）
    this.ctx.middleware(YGOProCtosResponse, async (msg: any, client: any, next: any) => {
      if (client.name !== NAME) return next();
      // 用对象引用精确区分「插件注入的」与「占位 WindBot 自己发的」
      if (this.injectedMsgs.has(msg)) return next();
      this.swallowed += 1;
      if (this.swallowed % 50 === 1) {
        this.logger.info({ swallowed: this.swallowed, from: client.name }, 'swallowed client response');
      }
      return undefined;
    }, true);

    // 2) 卡牌事件 + 连锁栈状态
    this.ctx.middleware(YGOProMsgBase, async (msg: any, client: any, next: any) => {
      const n = msg.constructor?.name ?? '';
      this.observation.observe(msg);
      // 默认关闭：每条 Move/Draw 都同步写盘，一局数百行
      if (process.env.MODEL_PLAYER_CARDEVENT !== '1') return next();
      if (/Summoned$|Chaining$|MsgSet$|MsgMove$|MsgDraw$|MsgWin$/.test(n)) {
        const loc = (o: any) => (o ? `${o.controller}/${o.location}/${o.sequence}` : undefined);
        this.logger.info(
          {
            msgName: n,
            code: msg.code,
            cards: Array.isArray(msg.cards) ? msg.cards : undefined,
            controller: msg.controller,
            player: msg.player,
            desc: msg.desc,
            chainCount: msg.chainCount,
            from: msg.previous ? loc(msg.previous) : undefined,
            to: msg.current ? loc(msg.current) : undefined,
          },
          'card event',
        );
      }
      return next();
    }, true);

    // 3) RETRY 诊断 + 保守重答
    this.ctx.middleware(YGOProMsgRetry, async (msg: any, client: any, next: any) => {
      const room = this.ctx.get(() => RoomManager).findByName(client.roomName) as any;
      if (room) {
        this.retries += 1;
        const req = room.lastResponseRequestMsg;
        const detail: any = {
          retries: this.retries,
          respPlayerName: room.responsePlayer?.name,
          accept: room.acceptResponse,
          lastReqKind: req?.constructor?.name,
          lastInjected: this.lastInjected,
        };
        if (req && typeof req === 'object') {
          // 记录请求里所有 count / canX 字段，定位合法候选数量
          for (const k of Object.keys(req)) {
            if (/count$|can[A-Z]/.test(k)) {
              const v = (req as any)[k];
              if (typeof v === 'number') detail[k] = v;
            }
          }
        }
        this.logger.warn(detail, 'core sent RETRY');
      }
      // 关键：RETRY 绝不能发给占位 WindBot（它的 OnPacket 遇 Retry 会抛异常退出）
      if (room && room.responsePlayer?.name === NAME) {
        const reqMsg = room.lastResponseRequestMsg;
        const operating = room.responsePlayer;
        if (reqMsg && operating) {
          setTimeout(() => {
            if (room.lastResponseRequestMsg !== reqMsg) return;
            const r = buildResponse(reqMsg);
            if (!r.payload || !r.payload.length) return;
            const retryMsg = new YGOProCtosResponse().fromPartial({ response: r.payload } as any);
            this.injectedMsgs.add(retryMsg as any);
            operating.modelInjecting = true;
            this.ctx
              .dispatch(retryMsg as any, operating as any)
              .then(() => {
                // RETRY 重答也是保守兜底，必须计入 byHow
                const k = `retry:${r.how}`;
                this.byHow.set(k, (this.byHow.get(k) ?? 0) + 1);
                this.logger.info(
                  { kind: reqMsg.constructor?.name, room: room.name, how: r.how },
                  'retry re-answer (conservative)',
                );
              })
              .catch(() => undefined)
              .finally(() => {
                operating.modelInjecting = false;
              });
          }, POLL_INTERVAL);
        }
      }
      return undefined;
    }, true);

    // 2.5) 连接层埋点：判断「中断是卡效果还是掉线」只能靠这两条
    this.ctx.middleware(OnRoomDuelStart, async (event: any, client: any, next: any) => {
      this.duelT0 = Date.now();
      this.observation = new DuelObservation();
      // 每局开始快照 byHow，DUEL_END 求差得到「本局」的响应来源分布
      this.byHowBase = new Map(this.byHow);
      this.logger.info(
        {
          kind: 'duel_start',
          room: event.room?.name,
          duel_stage: event.room?.duelStage,
          injected: this.injected,
          pred_slow: this.predSlow,
          pred_empty: this.predEmpty,
          pred_fail: this.predFail,
        },
        'DUEL_START',
      );
      return next();
    });

    this.ctx.middleware(OnRoomLeavePlayer, async (event: any, client: any, next: any) => {
      this.logger.warn(
        {
          kind: 'player_leave',
          reason: event.reason,
          old_pos: event.oldPos,
          by_system: event.bySystem,
          others_disconnected: event.otherDisconnectedCount,
          room: event.room?.name,
          duel_stage: event.room?.duelStage,
          duel_ms: this.duelT0 ? Date.now() - this.duelT0 : -1,
          injected: this.injected,
          stale: this.stale,
          retries: this.retries,
        },
        'PLAYER_LEAVE',
      );
      return next();
    });

    // 3) 对局结束：DUEL_END 埋点 + 导出 yrp 录像
    this.ctx.middleware(OnRoomWin, async (event: any, client: any, next: any) => {
      const ret = await next();
      try {
        const rec0 = event.room?.lastDuelRecord;
        const winMsg0 = event.winMsg;
        const base = this.byHowBase ?? new Map<string, number>();
        let consN = 0;
        const consHow: Record<string, number> = {};
        const modelHow: Record<string, number> = {};
        for (const [k, v] of this.byHow) {
          const d = v - (base.get(k) ?? 0);
          if (d <= 0) continue;
          if (k.startsWith('model:')) {
            const head = k.split(':')[0];
            modelHow[head] = (modelHow[head] ?? 0) + d;
          } else {
            consN += d;
            consHow[k] = (consHow[k] ?? 0) + d;
          }
        }
        this.logger.info(
          {
            kind: 'duel_end',
            winner_name: client?.name ?? '?',
            winner_duel_pos: winMsg0?.player ?? -1,
            reason: winMsg0?.type ?? -1,
            win_match: !!event.winMatch,
            swapped: !!event.wasSwapped,
            responses: rec0?.responses?.length ?? -1,
            messages: rec0?.messages?.length ?? -1,
            retries: this.retries,
            stale: this.stale,
            swallowed: this.swallowed,
            cons: consN,
            // 用 k=v,k=v 而不是 JSON：pino-pretty 里嵌套引号会把日志切成多行
            cons_how: Object.entries(consHow).map(([k, v]) => `${k}=${v}`).join(','),
            // 决策构成：announce 是否走链上卡名、多选是否尊重 -1、素材/连锁发动次数
            ann_by_chain: this.annByChain,
            ann_no_chain: this.annNoChain,
            ann_model: this.annModel,
            neg1_stop: this.neg1Stop,
            sum_used: this.sumUsed,
            chain_act: this.chainAct,
            pred_fail: this.predFail,
            pred_slow: this.predSlow,
            pred_empty: this.predEmpty,
          },
          'DUEL_END',
        );
      } catch {
        // 埋点失败不能影响对局收尾
      }
      // 每局结束重建 ygoinf 会话，避免 rstate / h_actions 跨局累积
      this.model.reset();
      if (process.env.MODEL_PLAYER_YRP !== '1') return ret;
      const room = event.room;
      if (!room) return ret;
      try {
        const record = room.lastDuelRecord;
        if (!record) {
          this.logger.warn({}, 'yrp skipped: no duel record');
          return ret;
        }
        // —— 配平校验 ——
        // 回放按 messages 里的请求顺序依次消费 responses。同一条决策消息可能被
        // dispatch 多次（实测 SelectChain 会），两者数量不等时回放会 "error occurs"。
        const msgs = record.messages;
        const countReq = () => msgs.filter((m: any) => m instanceof YGOProMsgResponseBase).length;
        const before = { total: msgs.length, req: countReq(), resp: record.responses.length };
        if (before.req !== before.resp) {
          const seen = new Set<any>();
          const dedup: any[] = [];
          for (const m of msgs) {
            if (seen.has(m)) continue;
            seen.add(m);
            dedup.push(m);
          }
          record.messages = dedup;
          this.logger.warn(
            { before, after: { total: dedup.length, req: countReq() }, removed: msgs.length - dedup.length },
            'yrp messages deduped',
          );
        } else {
          this.logger.info(before, 'yrp balance ok');
        }
        // 逐条配对诊断 + 超长响应检查（>255 或 0 会被 writer 静默丢弃）
        {
          const reqMsgs = record.messages.filter((m: any) => m instanceof YGOProMsgResponseBase);
          const pairs = reqMsgs.slice(0, 80).map((m: any, i: number) => {
            const kind = (m.constructor?.name ?? '').replace('YGOProMsg', '');
            const r = record.responses[i];
            const pl = m.player ?? m.controller ?? '?';
            return `${i}|${kind}|p${pl}|${r ? Buffer.from(r).toString('hex') : 'MISSING'}`;
          });
          const lens = record.responses.map((b: any) => b.length);
          const overs = lens
            .map((l: number, i: number) => ({ i, l }))
            .filter((x: any) => x.l > 255 || x.l === 0);
          this.logger.info(
            {
              count: reqMsgs.length,
              respCount: lens.length,
              maxLen: lens.length ? Math.max(...lens) : 0,
              dropped: overs.length,
              droppedDetail: overs.slice(0, 10),
              seed: record.seed,
              startTime: record.startTime,
              players: record.players?.map((p: any) => p?.name),
              pairs,
            },
            'yrp response pairs',
          );
        }
        const yrpObj = record.toYrp({ hostinfo: room.hostinfo, isTag: false });
        // 与训练侧一致：去掉压缩位，headerVersion=1（否则 8 元素 seedSequence 不生效）
        const hdr = yrpObj.header;
        if (hdr) {
          hdr.flag &= ~0x1;
          hdr.headerVersion = 1;
        }
        const buf = Buffer.from(yrpObj.toYrp());
        // 修正 dataSize：C++ 写的是「纯 responses 段大小」
        {
          const players = record.players ?? [];
          const deckLen = (p: any) =>
            4 + (p?.deck?.main?.length ?? 0) * 4 + 4 + (p?.deck?.extra?.length ?? 0) * 4;
          const responsesStart = 80 + 40 * 2 + 16 + deckLen(players[0]) + deckLen(players[1]);
          const realSize = buf.length - responsesStart;
          const oldSize = buf.readUInt32LE(0x10);
          buf.writeUInt32LE(realSize, 0x10);
          this.logger.info({ oldSize, realSize, responsesStart, total: buf.length }, 'yrp dataSize fixed');
        }
        const dir = path.resolve(process.env.MODEL_PLAYER_YRP_DIR ?? 'replays');
        fs.mkdirSync(dir, { recursive: true });
        const stamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
        const file = path.join(dir, `${stamp}.yrp`);
        fs.writeFileSync(file, buf);
        this.logger.info({ file, size: buf.length }, 'yrp saved');
      } catch (e) {
        this.logger.warn({ error: e instanceof Error ? e.message : String(e) }, 'yrp save failed');
      }
      return ret;
    }, true);

    // 4) 拦截决策，延时注入
    this.ctx.middleware(YGOProMsgResponseBase, async (msg: any, client: any, next: any) => {
      const room = this.ctx.get(() => RoomManager).findByName(client.roomName) as any;
      if (!room) return next();
      const ret = await next();
      if (!TAKEOVER) return ret;
      const kind = msg.constructor?.name ?? 'unknown';
      this.kinds.set(kind, (this.kinds.get(kind) ?? 0) + 1);

      // 只读诊断：记录非模型玩家的 idlecmd 候选数
      if (client.name !== NAME && kind === 'YGOProMsgSelectIdleCmd') {
        this.oppIdle += 1;
        if (this.oppIdle <= 30) {
          const keys = [
            'summonableCount', 'spSummonableCount', 'reposableCount', 'msetableCount',
            'ssetableCount', 'activatableCount', 'canBp', 'canEp', 'canShuffle',
          ];
          const counts: any = {};
          for (const k of keys) if (typeof msg[k] === 'number') counts[k] = msg[k];
          this.logger.info({ n: this.oppIdle, player: client.name, counts }, 'opp idlecmd candidates');
        }
      }

      const deadline = Date.now() + POLL_TIMEOUT;
      let payload: Uint8Array | undefined;
      let how = 'none';

      const attempt = async () => {
        // 必须确认「当前等待回应的请求」仍是这条消息，否则会把上一条的响应错配给下一条
        if (room.lastResponseRequestMsg !== msg) {
          this.stale += 1;
          if (this.stale <= 10 || this.stale % 100 === 0) {
            this.logger.warn(
              { kind, stale: this.stale, nowReqKind: room.lastResponseRequestMsg?.constructor?.name },
              'stale inject skipped',
            );
          }
          return;
        }
        // 必须在延时之后取：next() 返回时 routeGameMsg 还没设置 responsePlayer
        const operating = room.responsePlayer;
        // MSG 的 controller、room.turnIngamePos、feat(6)/(7) 用的都是【ingame duel pos】；
        // operating.pos 是房间座位号。isPosSwapped 的局两者相反，直接用会让
        // is_first / is_my_turn 整体反转，并把全场卡身份镜像。
        const seated = operating ? room.getIngameDuelPos(operating) : -1;
        const diag = {
          kind,
          accept: room.acceptResponse,
          respPos: room.responsePos,
          respPlayerName: operating?.name,
          // duelPos = 房间座位（不随每局换边变），ingamePos = 本局实际座位
          duelPos: operating ? room.getDuelPos(operating) : -1,
          ingamePos: seated,
          swapped: !!room.isPosSwapped,
        };
        if (!operating || operating.name !== NAME) {
          this.logger.warn(diag, 'not model turn');
          return;
        }
        this.logger.info(diag, 'inject check');
        // routeGameMsg 尚未完成时继续轮询，而不是盲等
        if (!room.acceptResponse) {
          if (Date.now() < deadline) {
            setTimeout(() => void attempt(), POLL_INTERVAL);
            return;
          }
          this.logger.warn(diag, 'inject skipped (timeout)');
          return;
        }

        // 快照验证：只打印，不影响注入
        if (process.env.MODEL_PLAYER_SNAPSHOT === '1' && this.snapshots < 20) {
          this.snapshots += 1;
          try {
            const snap = await buildFieldSnapshot(room, seated);
            this.logger.info(
              {
                n: this.snapshots,
                kind,
                cards: snap.cards.length,
                lp: [snap.myLp, snap.opLp],
                turn: snap.turn,
                phase: snap.phase,
                byLoc: snap.cards.reduce((a: any, c: any) => {
                  a[c.location] = (a[c.location] ?? 0) + 1;
                  return a;
                }, {}),
                byCtrl: snap.cards.reduce((a: any, c: any) => {
                  a[c.controller] = (a[c.controller] ?? 0) + 1;
                  return a;
                }, {}),
                monsters: snap.cards
                  .filter((c: any) => c.location === 'mzone')
                  .map((c: any) => ({
                    code: c.code,
                    ctrl: c.controller,
                    seq: c.sequence,
                    atk: c.attack,
                    def: c.defense,
                    pos: c.position,
                    lv: c.level,
                  })),
              },
              'field snapshot',
            );
          } catch (e) {
            this.logger.warn({ error: e instanceof Error ? e.message : String(e) }, 'snapshot failed');
          }
        }

        // 多卡选择：响应是 [count][idx0][idx1]...，features.py 用 msg.selected 支持多轮 predict。
        // SelectUnselectCard 的 prepareResponse 是【单个】参数，不能放进来
        // （传数组 → 非法响应 → RETRY → 占位客户端崩溃）。
        const MULTI_CARD = /SelectCard|SelectTribute/;
        // SelectSum 走单独的逐轮预测分支；SelectDisfield 的位置集合编码仍旁路
        const SKIP_MODEL = /SelectDisfield/;

        if (this.observation.automaticDiscard(msg,room.phase)) {
          const choices = msg.cards.map((_: unknown, i: number) => i);
          for (let i=0; i<msg.min; i++) {
            const j=randomInt(i,choices.length);
            [choices[i],choices[j]]=[choices[j],choices[i]];
          }
          payload=msg.prepareResponse(choices.slice(0,msg.min).map((index:number) => IndexResponse(index)));
          how='auto:end-phase-discard';
        } else if (MODEL_URL && !SKIP_MODEL.test(kind)) {
          try {
            const pos = seated;
            if (await this.model.ensureDuel()) {
              const revealed = this.observation.context(pos, kind).revealed;
              const snap = await buildFieldSnapshot(room, pos, revealed);
              const preview = await this.observation.negationPreview(snap, msg, pos,
                offer => room.ocgcore.queryNegationPreview(offer)).catch(error => {
                  this.logger.warn({ error: String(error) }, 'target preview unavailable');
                  return {};
                });
              const mkInput = (selected: number[]) => this.observation.input(snap, msg, pos, selected, preview);
              if (this.chainLogged < 8) {
                this.chainLogged += 1;
                this.logger.info({n: this.chainLogged, kind, chain_len: this.observation.chainStack.length,
                  forced: this.observation.chainForced, spe: this.observation.speCount}, 'chain_info');
              }

              if (MULTI_CARD.test(kind)) {
                // min 可能是 0（「一张都不选也合法」）。以前 max(1,·) 把 min=0 抬成 1，
                // 模型永远没有「选 0 张」的出口，与训练侧新增的 finish 候选不一致。
                const need = Math.max(0, msg.min ?? 1);
                // cap 防爆：max 可能是 99（「选任意数量」），逐轮预测每次都是一次 HTTP
                const cap = Math.min(24, Math.max(need, msg.max ?? need));
                const picked: number[] = [];
                for (let r = 0; r < cap; r += 1) {
                  const input = mkInput(picked);
                  if (!input) break;
                  // 取完整候选表而不是 top1：features.py::add_skipped_back 会把
                  // 「已选过的卡 / 结束选择」以 -1 插回 probs
                  const ranked = await this.predictRankedTimed(kind, input);
                  if (!ranked || !ranked.length) break;
                  // 模型首选「结束选择」(-1 占位)：已选够 need 就必须尊重它，
                  // 否则 min<max 的多选（白雪 7 选、威凯特② 1~2 素材）会被强行选满
                  if (ranked[0].response < 0 && picked.length >= need) {
                    this.model.commit(ranked[0].at);
                    this.neg1Stop += 1;
                    break;
                  }
                  let take: any;
                  for (const c of ranked) {
                    if (c.prob < 0 || c.response < 0) continue;
                    if (picked.includes(c.response)) continue;
                    take = c;
                    break;
                  }
                  if (!take) break;
                  // 实际采纳的不是 top1，历史动作必须按采纳项记
                  this.model.commit(take.at);
                  picked.push(take.response);
                  // 选满 need 只是下限，必须继续问到 cap，否则 min<max 永远只选 1 个
                  if (picked.length >= cap) break;
                }
                // need == 0 时允许空列表（选 0 张）；need >= 1 时空列表不满足条件，
                // 不会把「一张没选」的错误响应发给 core。
                if (picked.length >= need && msg.prepareResponse) {
                  payload = msg.prepareResponse(picked.map((i: number) => IndexResponse(i)));
                  how = `model:card[${picked.join(',')}]`;
                }
              } else if (kind === 'YGOProMsgSelectSum') {
                // select_sum 的协议响应是 [count][idx...]（SelectSum.prepareResponse →
                // Uint8Array(1+n)），不是 4 字节 int32，不能走 responseToBytes；
                // core 每次只接受「组合里下一张卡」的索引，必须逐轮预测 ——
                // 与 ygoenv 的 ms_mode_==1 / _callback_multi_select_2 对齐。
                const picked: number[] = [];
                let canFinish = false;
                const cap = Math.min(8, Math.max(1, msg.max ?? 1));
                for (let r = 0; r < cap; r += 1) {
                  const input = mkInput(picked);
                  if (!input) break;
                  const pred = await this.predictTimed(kind, input);
                  if (!pred) break;
                  // features.py 的 select_sum 候选 response 就是卡索引
                  if (pred.response < 0) break;
                  picked.push(pred.response);
                  // features.py: can_finish = 该卡单独就能补完组合；
                  // 训练侧组合耗尽即自动结束，这里必须同样立刻停
                  if (pred.canFinish) {
                    canFinish = true;
                    break;
                  }
                }
                // min == 0 时「一张都不选」是合法响应（空列表），不能被跳过；
                // min >= 1 时空列表仍走保守兜底，避免发出非法响应。
                const sumMin = msg.min ?? 1;
                if ((picked.length > 0 || sumMin === 0)
                  && typeof msg.prepareResponse === 'function') {
                  payload = msg.prepareResponse(picked.map((i: number) => IndexResponse(i)));
                  how = `model:sum[${picked.join(',')}]${canFinish ? '' : ':incomplete'}`;
                  this.sumUsed += 1;
                }
              } else {
                const input = mkInput([]);
                if (input) {
                  const pred = await this.predictTimed(kind, input);
                  if (pred) {
                    payload = responseToBytes(msg, pred.response);
                    how = `model:${pred.response}:${pred.prob.toFixed(3)}`;
                    if (kind === 'YGOProMsgAnnounceCard') this.annModel += 1;
                    if (kind === 'YGOProMsgSelectChain' && pred.response >= 0) this.chainAct += 1;
                  }
                }
              }
            }
          } catch (e) {
            this.logger.warn({ kind, error: e instanceof Error ? e.message : String(e) }, 'model path threw');
          }
        }

        if (!payload || payload.length === 0) {
          try {
            const r = buildResponse(msg);
            payload = r.payload;
            how = r.how;
          } catch (e) {
            this.logger.warn({ kind, error: e instanceof Error ? e.message : String(e) }, 'buildResponse threw');
          }
        }
        if (!payload || payload.length === 0) {
          this.logger.warn({ kind, how }, 'no response built, skip');
          return;
        }

        if (kind === 'YGOProMsgSelectCard' || (kind === 'YGOProMsgSelectBattleCmd' && payload.length === 4)) {
          const response = kind === 'YGOProMsgSelectBattleCmd'
            ? new DataView(payload.buffer,payload.byteOffset,4).getUint32(0,true) : 0;
          this.observation.committed(msg,response);
        }
        this.byHow.set(how, (this.byHow.get(how) ?? 0) + 1);
        this.injected += 1;
        const respHex = Buffer.from(payload).toString('hex');
        this.lastInjected = `${kind} / ${how} / ${respHex}`;
        const respMsg = new YGOProCtosResponse().fromPartial({ response: payload } as any);
        this.injectedMsgs.add(respMsg as any);
        operating.modelInjecting = true;
        this.ctx
          .dispatch(respMsg as any, operating as any)
          .then(() => {
            // SelectChain 量最大（44 局 >4200 次），此前 1/20 采样把「模型真的
            // 选择发动连锁」也一起埋掉了。现在：pass(-1) 仍采样，真正发动全量记录。
            const important =
              !/SelectChain/.test(kind) ||
              this.injected % 20 === 0 ||
              /^model:(?!-)/.test(how ?? '');
            if (important) {
              // seat/ingamePos 必须一起记：两者不同说明本局换边了
              this.logger.info(
                {
                  injected: this.injected,
                  room: client.roomName,
                  kind,
                  how,
                  resp: respHex,
                  seat: room.getDuelPos(operating),
                  ingamePos: room.getIngameDuelPos(operating),
                  swapped: !!room.isPosSwapped,
                },
                'inject ok',
              );
            }
          })
          .catch((e: any) =>
            this.logger.warn({ kind, error: e instanceof Error ? e.message : String(e) }, 'inject failed'),
          )
          .finally(() => {
            operating.modelInjecting = false;
          });
      };

      setTimeout(attempt, POLL_INTERVAL);
      return ret;
    }, true);
  }
}
