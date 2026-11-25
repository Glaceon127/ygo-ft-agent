// 保守响应构造器：全部基于 ygopro-msg-encode 的官方 API
//   1) 优先 defaultResponse()（仅 7 个类实现：chain/card/yesno/effectyn/tribute/unselect/sort）
//   2) 其余用各自的 prepareResponse(...)（每个消息类都有）
//   3) 参数一律取该消息里 core 给出的合法候选，不硬编码字节
import {
  YGOProMsgResponseBase,
  YGOProMsgSelectIdleCmd,
  YGOProMsgSelectBattleCmd,
  YGOProMsgSelectPlace,
  YGOProMsgSelectPosition,
  YGOProMsgSelectOption,
  YGOProMsgSelectSum,
  YGOProMsgSelectCard,
  YGOProMsgSelectTribute,
  YGOProMsgSelectUnselectCard,
  YGOProMsgSelectChain,
  YGOProMsgSelectYesNo,
  YGOProMsgSelectEffectYn,
  YGOProMsgSelectCounter,
  YGOProMsgSortCard,
  YGOProMsgAnnounceNumber,
  YGOProMsgAnnounceAttrib,
  YGOProMsgAnnounceRace,
  YGOProMsgAnnounceCard,
  IdleCmdType,
  BattleCmdType,
  IndexResponse,
} from 'ygopro-msg-encode';
import { parseAnnounceCardCodes } from './action-builder';

const POSITIONS = [0x1, 0x2, 0x4, 0x8];

// set_responsei 的整数响应：回放按 4 字节读（ygopro.h::YGO_SetResponsei）
function i32(v: number): Uint8Array {
  const b = new Uint8Array(4);
  new DataView(b.buffer).setInt32(0, v, true);
  return b;
}

export interface BuildResult {
  payload?: Uint8Array;
  how: string;
}

/** 取位掩码里第一个可用的 position 位 */
function firstPosition(mask: number): number | undefined {
  for (const p of POSITIONS) {
    if (mask & p) return p;
  }
  return undefined;
}

export function buildResponse(msg: YGOProMsgResponseBase): BuildResult {
  // ---------- 1. 官方默认响应 ----------
  try {
    const d = msg.defaultResponse();
    if (d && d.length) return { payload: d, how: 'defaultResponse' };
  } catch {
    // 继续走 prepareResponse
  }

  // ---------- 2. 主阶段 ----------
  if (msg instanceof YGOProMsgSelectIdleCmd) {
    if (msg.canEp) {
      return {
        payload: msg.prepareResponse(IdleCmdType.TO_EP),
        how: 'idlecmd:TO_EP',
      };
    }
    if (msg.canBp) {
      return {
        payload: msg.prepareResponse(IdleCmdType.TO_BP),
        how: 'idlecmd:TO_BP',
      };
    }
    const groups: Array<[IdleCmdType, Array<Record<string, unknown>>]> = [
      [IdleCmdType.SUMMON, msg.summonableCards as never],
      [IdleCmdType.SPSUMMON, msg.spSummonableCards as never],
      [IdleCmdType.MSET, msg.msetableCards as never],
      [IdleCmdType.ACTIVATE, msg.activatableCards as never],
      [IdleCmdType.REPOS, msg.reposableCards as never],
      [IdleCmdType.SSET, msg.ssetableCards as never],
    ];
    for (const [type, cards] of groups) {
      const c = cards?.[0];
      if (c) {
        return {
          payload: msg.prepareResponse(type, {
            code: c.code as number,
            controller: c.controller as number,
            location: c.location as number,
            sequence: c.sequence as number,
            desc: c.desc as number | undefined,
          }),
          how: `idlecmd:${IdleCmdType[type]}`,
        };
      }
    }
    if (msg.canShuffle) {
      return {
        payload: msg.prepareResponse(IdleCmdType.SHUFFLE),
        how: 'idlecmd:SHUFFLE',
      };
    }
    // 兜底：绝不能留空。留空会让 core 等不到响应，对局直接卡死（MISSING）。
    try {
      const d = msg.defaultResponse?.();
      if (d && d.length) {
        return { payload: d, how: 'idlecmd:default' };
      }
    } catch {
      // 继续尝试强制结束回合
    }
    return {
      payload: msg.prepareResponse(IdleCmdType.TO_EP),
      how: 'idlecmd:fallback-ep',
    };
  }

  // ---------- 3. 战斗阶段 ----------
  if (msg instanceof YGOProMsgSelectBattleCmd) {
    if (msg.canEp) {
      return {
        payload: msg.prepareResponse(BattleCmdType.TO_EP),
        how: 'battlecmd:TO_EP',
      };
    }
    if (msg.canM2) {
      return {
        payload: msg.prepareResponse(BattleCmdType.TO_M2),
        how: 'battlecmd:TO_M2',
      };
    }
    const atk = msg.attackableCards?.[0];
    if (atk) {
      return {
        payload: msg.prepareResponse(BattleCmdType.ATTACK, {
          code: atk.code,
          controller: atk.controller,
          location: atk.location,
          sequence: atk.sequence,
        }),
        how: 'battlecmd:ATTACK',
      };
    }
    const act = msg.activatableCards?.[0];
    if (act) {
      return {
        payload: msg.prepareResponse(BattleCmdType.ACTIVATE, {
          code: act.code,
          controller: act.controller,
          location: act.location,
          sequence: act.sequence,
          desc: act.desc,
        }),
        how: 'battlecmd:ACTIVATE',
      };
    }
    return { how: 'battlecmd:none' };
  }

  // ---------- 4. 摆放 ----------
  if (msg instanceof YGOProMsgSelectPlace) {
    const places = msg.getSelectablePlaces();
    if (places?.length) {
      const need = Math.max(1, msg.count ?? 1);
      const picked = places.slice(0, need).map((p) => ({
        player: p.player,
        location: p.location,
        sequence: p.sequence,
      }));
      return {
        payload: msg.prepareResponse(picked),
        how: `place:${picked.length}`,
      };
    }
    return { how: 'place:none' };
  }

  // ---------- 5. 表示形式 ----------
  if (msg instanceof YGOProMsgSelectPosition) {
    const p = firstPosition(msg.positions);
    if (p) {
      // set_responsei → 必须 4 字节（回放格式）
        return { payload: i32(p), how: `position:0x${p.toString(16)}` };
    }
    return { how: 'position:none' };
  }

  // ---------- 6. 选项 ----------
  if (msg instanceof YGOProMsgSelectOption) {
    const count = msg.count ?? msg.options?.length ?? 0;
    if (count > 0) {
      return {
        payload: msg.prepareResponse(IndexResponse(0)),
        how: 'option:0',
      };
    }
    return { how: 'option:none' };
  }

  // ---------- 7. 选卡类 ----------
  if (msg instanceof YGOProMsgSelectCard) {
    // min>0 时必须选够；取前 min 张（models.py: min=0 不支持）
    const total = msg.cards?.length ?? 0;
    const need = Math.max(1, Math.min(msg.min ?? 1, total));
    const picks = Array.from({ length: need }, (_, i) => IndexResponse(i));
    return { payload: msg.prepareResponse(picks), how: `card:first${need}` };
  }

  if (msg instanceof YGOProMsgSelectTribute) {
    const total = msg.cards?.length ?? 0;
    const need = Math.max(1, Math.min(msg.min ?? 1, total));
    const picks = Array.from({ length: need }, (_, i) => IndexResponse(i));
    return { payload: msg.prepareResponse(picks), how: `tribute:first${need}` };
  }

  if (msg instanceof YGOProMsgSelectUnselectCard) {
    return { payload: msg.prepareResponse(), how: 'unselect:default' };
  }

  if (msg instanceof YGOProMsgSelectSum) {
    // 关键：SelectSum.prepareResponse 会自己把 mustSelectCount 写进 buffer[0] 前缀，
    // 参数只能是 msg.cards 里的索引。以前把 mustSelectCards 也当成 CardInfo 传进去，
    // findIndex 在 this.cards 里找不到 → 抛 "Card not found" → payload 为空 →
    // core 永远等不到响应，对局卡死到超时。
    const total = msg.cards?.length ?? 0;
    if (total === 0) {
      // 只有强制项，没有可选项：[mustSelectCount] 就是合法响应
      return { payload: msg.prepareResponse([]), how: 'sum:must-only' };
    }
    const need = Math.max(1, Math.min(msg.min ?? 1, total));
    const picks = Array.from({ length: need }, (_, i) => IndexResponse(i));
    return { payload: msg.prepareResponse(picks), how: `sum:first${need}` };
  }

  if (msg instanceof YGOProMsgSelectChain) {
    return { payload: msg.prepareResponse(), how: 'chain:default' };
  }

  if (msg instanceof YGOProMsgSelectYesNo) {
    return { payload: i32(0), how: 'yesno:false' };
  }

  if (msg instanceof YGOProMsgSelectEffectYn) {
    return { payload: i32(0), how: 'effectyn:false' };
  }

  if (msg instanceof YGOProMsgSelectCounter) {
    // 源码签名: prepareResponse(Array<{ card: IndexResponseObject | CardInfo, count: number }>)
    let left = msg.counterCount;
    const opts = msg.cards.map((c, i) => {
      const count = Math.min(left, c.counterCount);
      left -= count;
      return { card: IndexResponse(i), count };
    });
    if (opts.length && left === 0) {
      return { payload: msg.prepareResponse(opts), how: 'counter:allocate' };
    }
    return { how: 'counter:none' };
  }

  if (msg instanceof YGOProMsgSortCard) {
    return { payload: msg.prepareResponse(), how: 'sort:default' };
  }

  // ---------- 8. 宣言类 ----------
  if (msg instanceof YGOProMsgAnnounceNumber) {
    return { payload: msg.prepareResponse(IndexResponse(0)), how: 'announce:number' };
  }

  if (msg instanceof YGOProMsgAnnounceAttrib) {
    return { payload: msg.prepareResponse(0x1), how: 'announce:attrib' };
  }

  if (msg instanceof YGOProMsgAnnounceRace) {
    return { payload: msg.prepareResponse(0x1), how: 'announce:race' };
  }

  if (msg instanceof YGOProMsgAnnounceCard) {
    // 以前读 msg.cards[0].code，但协议层根本没有 cards 字段 → 恒回退 14558127，
    // 实测 9 次全被固定宣言成灰流丽。改成从 opcodes 解析 core 给出的合法候选。
    const codes = parseAnnounceCardCodes(msg);
    const code = codes[0] ?? 14558127; // 解析失败时保底，避免 payload 为空卡死 core
    return { payload: msg.prepareResponse(code), how: `announce:card:${code}` };
  }

  return { how: 'unhandled' };
}
