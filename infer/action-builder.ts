// YGOProMsgSelect* → ygoinf 的 action_msg
//
// 核心策略：凡需要 `response` 字段的地方，一律用官方
//   prepareResponse(IndexResponse(i)) / prepareResponse(type, IndexResponse(i))
// 生成后读回 uint32，而不是自己拼位运算。
// 依据：index.mjs:5108  region  view.setUint32(0, sequence << 16 | type, true)
import {
  YGOProMsgResponseBase,
  YGOProMsgSelectIdleCmd,
  YGOProMsgSelectBattleCmd,
  YGOProMsgSelectChain,
  YGOProMsgSelectCard,
  YGOProMsgSelectTribute,
  YGOProMsgSelectSum,
  YGOProMsgSelectUnselectCard,
  YGOProMsgSelectPosition,
  YGOProMsgSelectPlace,
  YGOProMsgSelectDisField,
  YGOProMsgSelectOption,
  YGOProMsgSelectYesNo,
  YGOProMsgSelectEffectYn,
  YGOProMsgAnnounceAttrib,
  YGOProMsgAnnounceNumber,
  YGOProMsgAnnounceCard,
  IdleCmdType,
  BattleCmdType,
  IndexResponse,
} from 'ygopro-msg-encode';

// ---- 基础字段映射（与 input-builder 保持一致）----
const LOC: Record<number, string> = {
  1: 'deck', 2: 'hand', 4: 'mzone', 8: 'szone', 16: 'grave',
  32: 'removed', 64: 'extra', 128: 'szone', 256: 'szone', 512: 'szone',
};

const IDLE_NAME: Record<number, string> = {
  0: 'summon', 1: 'sp_summon', 2: 'reposition', 3: 'mset',
  4: 'set', 5: 'activate', 6: 'to_bp', 7: 'to_ep', 8: 'to_ep',
};

const BATTLE_NAME: Record<number, string> = {
  0: 'activate', 1: 'attack', 2: 'to_m2', 3: 'to_ep',
};

const ATTR_BITS: Array<[number, string]> = [
  [1, 'earth'], [2, 'water'], [4, 'fire'], [8, 'wind'],
  [16, 'light'], [32, 'dark'], [64, 'divine'],
];

const POS_BITS: Array<[number, string]> = [
  [1, 'faceup_attack'], [2, 'facedown_attack'],
  [4, 'faceup_defense'], [8, 'facedown_defense'],
];

function u32(bytes: Uint8Array | undefined): number {
  if (!bytes || bytes.length < 4) return 0;
  return new DataView(bytes.buffer, bytes.byteOffset, 4).getUint32(0, true);
}

function ctrl(controller: number, myDuelPos: number): string {
  return controller === myDuelPos ? 'me' : 'opponent';
}

function loc(l: number | undefined): string {
  return LOC[l ?? 0] ?? 'szone';
}

/** models.py CardLocation */
function cardLocation(
  controller: number | undefined,
  location: number | undefined,
  sequence: number | undefined,
  myDuelPos: number,
) {
  return {
    controller: ctrl(controller ?? 0, myDuelPos),
    location: loc(location),
    sequence: sequence ?? 0,
    // models.py: overlay 索引从 0 起，非 overlay 为 -1
    overlay_sequence: -1,
  };
}

/** 所有消息都带 msg_type，先取出来 */
function msgType(msg: YGOProMsgResponseBase): string {
  return (msg as unknown as { msg_type?: string }).msg_type ?? '';
}

export function buildActionMsg(
  msg: YGOProMsgResponseBase,
  myDuelPos: number,
  selectedIdx: number[] = [],
): Record<string, unknown> | undefined {
  // ---------------- 主阶段 ----------------
  if (msg instanceof YGOProMsgSelectIdleCmd) {
    const idle: Array<Record<string, unknown>> = [];
    const groups: Array<[IdleCmdType, Array<Record<string, number>>]> = [
      [IdleCmdType.SUMMON, msg.summonableCards as never],
      [IdleCmdType.SPSUMMON, msg.spSummonableCards as never],
      [IdleCmdType.REPOS, msg.reposableCards as never],
      [IdleCmdType.MSET, msg.msetableCards as never],
      [IdleCmdType.SSET, msg.ssetableCards as never],
      [IdleCmdType.ACTIVATE, msg.activatableCards as never],
    ];
    for (const [type, cards] of groups) {
      for (let i = 0; i < (cards?.length ?? 0); i += 1) {
        const c = cards[i];
        idle.push({
          cmd_type: IDLE_NAME[type],
          data: {
            card_info: {
              code: c.code ?? 0,
              ...cardLocation(c.controller, c.location, c.sequence, myDuelPos),
            },
            effect_description: c.desc ?? 0,
            response: u32(msg.prepareResponse(type, IndexResponse(i))),
          },
        });
      }
    }
    // TO_BP / TO_EP 无 data（models.py: IdleCmd.data 可选）
    if (msg.canBp) idle.push({ cmd_type: 'to_bp', data: null });
    if (msg.canEp) idle.push({ cmd_type: 'to_ep', data: null });
    return { msg_type: 'select_idlecmd', idle_cmds: idle };
  }

  // ---------------- 战斗阶段 ----------------
  if (msg instanceof YGOProMsgSelectBattleCmd) {
    const cmds: Array<Record<string, unknown>> = [];
    for (let i = 0; i < (msg.activatableCards?.length ?? 0); i += 1) {
      const c = msg.activatableCards[i];
      cmds.push({
        cmd_type: BATTLE_NAME[BattleCmdType.ACTIVATE],
        data: {
          card_info: {
            code: c.code ?? 0,
            ...cardLocation(c.controller, c.location, c.sequence, myDuelPos),
          },
          effect_description: c.desc ?? 0,
          direct_attackable: false,
          response: u32(msg.prepareResponse(BattleCmdType.ACTIVATE, IndexResponse(i))),
        },
      });
    }
    for (let i = 0; i < (msg.attackableCards?.length ?? 0); i += 1) {
      const c = msg.attackableCards[i];
      cmds.push({
        cmd_type: BATTLE_NAME[BattleCmdType.ATTACK],
        data: {
          card_info: {
            code: c.code ?? 0,
            ...cardLocation(c.controller, c.location, c.sequence, myDuelPos),
          },
          effect_description: 0,
          direct_attackable: !!c.directAttack,
          response: u32(msg.prepareResponse(BattleCmdType.ATTACK, IndexResponse(i))),
        },
      });
    }
    if (msg.canM2) cmds.push({ cmd_type: 'to_m2', data: null });
    if (msg.canEp) cmds.push({ cmd_type: 'to_ep', data: null });
    return { msg_type: 'select_battlecmd', battle_cmds: cmds };
  }

  // ---------------- 连锁 ----------------
  if (msg instanceof YGOProMsgSelectChain) {
    const chains: Array<Record<string, unknown>> = [];
    for (let i = 0; i < (msg.chains?.length ?? 0); i += 1) {
      const c = msg.chains[i];
      chains.push({
        code: c.code ?? 0,
        location: cardLocation(c.controller, c.location, c.sequence, myDuelPos),
        effect_description: c.desc ?? c.edesc ?? 0,
        response: u32(msg.prepareResponse(IndexResponse(i))),
      });
    }
    return {
      msg_type: 'select_chain',
      // specialCount > 0 表示存在强制项
      forced: (msg.specialCount ?? 0) > 0,
      chains,
    };
  }

  // ---------------- 选卡 ----------------
  // 注意：select_card 的协议响应是 [count][idx0][idx1]...（index.mjs:4754），
  // 不是 4 字节位置编码。所以这里的 response 统一用「卡索引」，
  // 由客户端最终 prepareResponse(IndexResponse(i)...) 生成。
  if (msg instanceof YGOProMsgSelectCard) {
    const cards = (msg.cards ?? []).map((c, i) => ({
      location: cardLocation(c.controller, c.location, c.sequence, myDuelPos),
      response: i,
    }));
    return {
      msg_type: 'select_card',
      cancelable: !!msg.cancelable,
      min: msg.min ?? 0,
      max: msg.max ?? 0,
      cards,
      selected: selectedIdx,
    };
  }

  if (msg instanceof YGOProMsgSelectTribute) {
    const cards = (msg.cards ?? []).map((c, i) => ({
      location: cardLocation(c.controller, c.location, c.sequence, myDuelPos),
      // ygopro-msg-encode 只给 releaseParam，没有 level；models.py 需要 level
      level: (c as unknown as { level?: number }).level ?? 0,
      response: i,
    }));
    return {
      msg_type: 'select_tribute',
      cancelable: !!msg.cancelable,
      min: msg.min ?? 0,
      max: msg.max ?? 0,
      cards,
      selected: selectedIdx,
    };
  }

  if (msg instanceof YGOProMsgSelectSum) {
    // select_sum 的协议响应是 [count][idx...]（index.mjs: SelectSum.prepareResponse
    // 返回 Uint8Array(1+n)），不是 4 字节 int32。以前这里用 u32(...) 取前 4 字节，
    // 但缓冲只有 2 字节 → u32 恒为 0，所有候选项 response 全一样，
    // 模型选出来再编码回去必然被 core 判 RETRY。这里和 select_card 一样统一用「卡索引」，
    // 由客户端 prepareResponse(IndexResponse(i)...) 生成真正的响应。
    const conv = (c: Record<string, number>, i: number, isMust: boolean) => ({
      location: cardLocation(c.controller, c.location, c.sequence, myDuelPos),
      // CardInfo.opParam 是 core 给的等级/数值参数
      level1: c.opParam ?? 0,
      level2: 0,
      // must 卡不参与选择，response 固定 0（与 features.py 一致）
      response: isMust ? 0 : i,
    });
    return {
      msg_type: 'select_sum',
      overflow: false,
      level_sum: msg.sumVal ?? 0,
      min: msg.min ?? 0,
      max: msg.max ?? 0,
      cards: (msg.cards ?? []).map((c, i) => conv(c as never, i, false)),
      must_cards: (msg.mustSelectCards ?? []).map((c, i) => conv(c as never, i, true)),
      // 逐轮预测必须把已选索引回传，features.py 用 selected 从组合里扣掉已选项
      selected: selectedIdx,
    };
  }

  if (msg instanceof YGOProMsgSelectUnselectCard) {
    // 同 select_sum：prepareResponse 返回 Uint8Array(2) = [1][index]，
    // u32 取 4 字节时缓冲不够 → 恒为 0，所有候选项 response 全相同，
    // 模型无论选谁最后都只能发出 index 0。这里统一用卡索引。
    const conv = (c: Record<string, number>, i: number) => ({
      location: cardLocation(c.controller, c.location, c.sequence, myDuelPos),
      response: i,
    });
    return {
      msg_type: 'select_unselect_card',
      finishable: !!msg.finishable,
      cancelable: !!msg.cancelable,
      min: msg.min ?? 0,
      max: msg.max ?? 0,
      selected_cards: [],
      selectable_cards: (msg.selectableCards ?? []).map((c, i) => conv(c as never, i)),
    };
  }

  // ---------------- 位置 / 摆放 ----------------
  if (msg instanceof YGOProMsgSelectPosition) {
    const positions: string[] = [];
    for (const [bit, name] of POS_BITS) {
      if ((msg.positions ?? 0) & bit) positions.push(name);
    }
    return {
      msg_type: 'select_position',
      code: msg.code ?? 0,
      positions: positions.length ? positions : ['none'],
    };
  }

  if (msg instanceof YGOProMsgSelectPlace) {
    const places = (msg.getSelectablePlaces() ?? []).slice(
      0,
      Math.max(1, msg.count ?? 1),
    ).map((p) => ({
      controller: ctrl(p.player, myDuelPos),
      location: loc(p.location),
      sequence: p.sequence,
    }));
    return { msg_type: 'select_place', count: msg.count ?? 1, places };
  }

  if (msg instanceof YGOProMsgSelectDisField) {
    const places = (msg.getSelectablePlaces() ?? []).slice(
      0,
      Math.max(1, msg.count ?? 1),
    ).map((p) => ({
      controller: ctrl(p.player, myDuelPos),
      location: loc(p.location),
      sequence: p.sequence,
    }));
    return { msg_type: 'select_disfield', count: msg.count ?? 1, places };
  }

  // ---------------- 选项 / 是非 ----------------
  if (msg instanceof YGOProMsgSelectOption) {
    const options = Array.from({ length: msg.count ?? 0 }, (_, i) => ({
      code: msg.options?.[i] ?? 0,
      response: u32(msg.prepareResponse(IndexResponse(i))),
    }));
    return { msg_type: 'select_option', options };
  }

  if (msg instanceof YGOProMsgSelectYesNo) {
    return {
      msg_type: 'select_yesno',
      effect_description: msg.desc ?? 0,
    };
  }

  if (msg instanceof YGOProMsgSelectEffectYn) {
    return {
      msg_type: 'select_effectyn',
      code: msg.code ?? 0,
      location: cardLocation(msg.controller, msg.location, msg.sequence, myDuelPos),
      effect_description: msg.desc ?? 0,
    };
  }

  // ---------------- 宣言 ----------------
  if (msg instanceof YGOProMsgAnnounceAttrib) {
    const attributes: Array<Record<string, unknown>> = [];
    for (const [bit, name] of ATTR_BITS) {
      if ((msg.availableAttributes ?? 0) & bit) {
        attributes.push({
          attribute: name,
          response: u32(msg.prepareResponse(bit)),
        });
      }
    }
    return { msg_type: 'announce_attrib', count: msg.count ?? 1, attributes };
  }

  if (msg instanceof YGOProMsgAnnounceNumber) {
    const numbers = (msg.numbers ?? []).map((n, i) => ({
      number: n,
      response: u32(msg.prepareResponse(IndexResponse(i))),
    }));
    return { msg_type: 'announce_number', count: msg.count ?? 1, numbers };
  }

  if (msg instanceof YGOProMsgAnnounceCard) {
    const codes = parseAnnounceCardCodes(msg);
    if (!codes.length) return undefined;
    const cards = codes.map((code) => ({
      code,
      response: u32(msg.prepareResponse(code)),
    }));
    // count=1 表示「宣言 1 个卡名」，不是 opcodes 的数组长度。
    return { msg_type: 'announce_card', count: 1, cards };
  }

  return undefined;
}

// MSG_ANNOUNCE_CARD 的 opcodes 解析，对齐 ygopro.h:153 parse_codes_from_opcodes。
// 协议层只有 player/count/opcodes，没有 cards 字段；候选卡号藏在 opcodes 里。
export function parseAnnounceCardCodes(msg: YGOProMsgAnnounceCard): number[] {
  const ops: number[] = (msg as unknown as { opcodes?: number[] }).opcodes ?? [];
  const n = ops.length;
  if (n === 2) return [ops[0]];
  if (n < 2 || (n - 2) % 3 !== 0) return [];
  const codes: number[] = [];
  for (let i = 2; i < n; i += 3) {
    if (ops[i + 1] !== 1073742080 || ops[i + 2] !== 1073741829) return [];
    codes.push(ops[i]);
  }
  return codes;
}

export { msgType };
