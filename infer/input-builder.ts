// 从 srvpro2 的 CardQuery 构造 ygoinf 的 Input
//
// 依据：
//   卡顺序    ygopro.h::_set_obs_g_cards
//             for player in [me, opp] -> for loc in DECK,HAND,MZONE,SZONE,GRAVE,REMOVED,EXTRA
//   字段语义  ygoinf/models.py 的 Card/Global 枚举（字符串逐字一致）
//   位掩码    ygopro-msg-encode/dist/src/vendor/ocgcore-constants.d.ts
//   +1/归一化 由 Python 侧 encode_card 处理，这里传原始值
import { Room } from '../../src/room';
import { readCardWithReader } from '../../src/utility';

// ---- 位掩码（数值直接取自 ocgcore-constants.d.ts）----
const LOC = {
  DECK: 1, HAND: 2, MZONE: 4, SZONE: 8, GRAVE: 16, REMOVED: 32,
  EXTRA: 64, OVERLAY: 128, FZONE: 256, PZONE: 512,
} as const;

const POS = {
  FACEUP_ATTACK: 1, FACEDOWN_ATTACK: 2, FACEUP_DEFENSE: 4, FACEDOWN_DEFENSE: 8,
} as const;

const ATTR: Array<[number, string]> = [
  [1, 'earth'], [2, 'water'], [4, 'fire'], [8, 'wind'],
  [16, 'light'], [32, 'dark'], [64, 'divine'],
];

const RACE: Array<[number, string]> = [
  [1, 'warrior'], [2, 'spellcaster'], [4, 'fairy'], [8, 'fiend'],
  [16, 'zombie'], [32, 'machine'], [64, 'aqua'], [128, 'pyro'],
  [256, 'rock'], [512, 'windbeast'], [1024, 'plant'], [2048, 'insect'],
  [4096, 'thunder'], [8192, 'dragon'], [16384, 'beast'], [32768, 'beast_warrior'],
  [65536, 'dinosaur'], [131072, 'fish'], [262144, 'sea_serpent'],
  [524288, 'reptile'], [1048576, 'psycho'], [2097152, 'devine'],
  [4194304, 'creator_god'], [8388608, 'wyrm'], [16777216, 'cyberse'],
  [33554432, 'illusion'],
];

const TYPE: Array<[number, string]> = [
  [1, 'monster'], [2, 'spell'], [4, 'trap'], [16, 'normal'], [32, 'effect'],
  [64, 'fusion'], [128, 'ritual'], [256, 'trap_monster'], [512, 'spirit'],
  [1024, 'union'], [2048, 'dual'], [4096, 'tuner'], [8192, 'synchro'],
  [16384, 'token'], [65536, 'quick_play'], [131072, 'continuous'],
  [262144, 'equip'], [524288, 'field'], [1048576, 'counter'],
  [2097152, 'flip'], [4194304, 'toon'], [8388608, 'xyz'],
  [16777216, 'pendulum'], [33554432, 'special'], [67108864, 'link'],
];

const STATUS_DISABLED = 1;
const STATUS_FORBIDDEN = 67108864;

// 与 _set_obs_g_cards 的 configs 顺序一致
const LOCATION_ORDER = [
  LOC.DECK, LOC.HAND, LOC.MZONE, LOC.SZONE, LOC.GRAVE, LOC.REMOVED, LOC.EXTRA,
];

// refresh-query.ts 的分区域 queryFlag
function zoneQueryFlag(location: number): number {
  switch (location) {
    case LOC.MZONE: return 0x891fff;
    case LOC.SZONE: return 0xe81fff;
    case LOC.HAND: return 0x681fff;
    case LOC.GRAVE: return 0x081fff;
    case LOC.EXTRA: return 0xe81fff;
    default: return 0xf81fff;
  }
}

function locationName(loc: number): string {
  switch (loc) {
    case LOC.DECK: return 'deck';
    case LOC.HAND: return 'hand';
    case LOC.MZONE: return 'mzone';
    case LOC.SZONE:
    case LOC.FZONE:
    case LOC.PZONE:
    case LOC.OVERLAY: return 'szone';
    case LOC.GRAVE: return 'grave';
    case LOC.REMOVED: return 'removed';
    case LOC.EXTRA: return 'extra';
    default: return 'szone';
  }
}

function positionName(pos: number | undefined): string {
  if (!pos) return 'none';
  if (pos === 5) return 'faceup';
  if (pos === 10) return 'facedown';
  if (pos === 3) return 'attack';
  if (pos === 12) return 'defense';
  if (pos & POS.FACEUP_ATTACK) return 'faceup_attack';
  if (pos & POS.FACEDOWN_ATTACK) return 'facedown_attack';
  if (pos & POS.FACEUP_DEFENSE) return 'faceup_defense';
  if (pos & POS.FACEDOWN_DEFENSE) return 'facedown_defense';
  return 'none';
}

function pickBit(value: number | undefined, table: Array<[number, string]>): string {
  if (!value) return 'none';
  for (const [bit, name] of table) {
    if (value & bit) return name;
  }
  return 'none';
}

function typeNames(t: number | undefined): string[] {
  if (!t) return [];
  const out: string[] = [];
  for (const [bit, name] of TYPE) {
    if (t & bit) out.push(name);
  }
  return out;
}

export interface YgoinfCard {
  code: number;
  public_prince_name?: boolean;
  location: string;
  sequence: number;
  controller: string;
  position: string;
  overlay_sequence: number;
  attribute: string;
  race: string;
  level: number;
  counter: number;
  negated: boolean;
  temporary?: boolean;
  attack: number;
  defense: number;
  types: string[];
}

export interface FieldSnapshot {
  opponentSpellCounters?: number;
  cards: YgoinfCard[];
  myLp: number;
  opLp: number;
  turn: number;
  phase: string;
  isFirst: boolean;
  isMyTurn: boolean;
}

// OCG 的 PHASE_* 是位掩码，不是序号（models.py Phase 枚举）
function phaseName(p: number | undefined): string {
  switch (p) {
    case 0x01: return 'draw';
    case 0x02: return 'standby';
    case 0x04: return 'main1';
    case 0x08: return 'battle_start';
    case 0x10: return 'battle_step';
    case 0x20: return 'damage';
    case 0x40: return 'damage_calculation';
    case 0x80: return 'battle';
    case 0x100: return 'main2';
    case 0x200: return 'end';
    default: return 'main1';
  }
}

/** 按 _set_obs_g_cards 的顺序抓全场地 */
export async function buildFieldSnapshot(
  room: Room,
  myDuelPos: number,
  revealed: string[] = [],
): Promise<FieldSnapshot> {
  const ocgcore = room.ocgcore;
  if (!ocgcore) throw new Error('no ocgcore');

  const cards: YgoinfCard[] = [];
  let opponentSpellCounters = 0;
  outer: for (const pi of [0, 1]) {
    const player = (myDuelPos + pi) % 2;
    const controller = pi === 0 ? 'me' : 'opponent';
    for (const loc of LOCATION_ORDER) {
      const res = await ocgcore.queryFieldCard({
        player,
        location: loc,
        queryFlag: zoneQueryFlag(loc) | 0x10000 | ((loc === LOC.MZONE || loc === LOC.SZONE) ? 0x20000 : 0),
        useCache: 0,
      });
      for (const q of res?.cards ?? []) {
        if (!q || q.empty) continue;
        const zone = q.location ?? loc;
        if (pi === 1 && (zone === LOC.MZONE || zone === LOC.SZONE) && ((q.position ?? 0) & 5)) {
          opponentSpellCounters += q.counters?.find(counter => counter.type === 1)?.count ?? 0;
        }
        const privateZone = zone === LOC.HAND || (!revealed.length && (zone === LOC.DECK || zone === LOC.EXTRA));
        if (!(pi === 1 && privateZone) && q.overlayCards?.length) {
          const reader = await room.getCardReader();
          for (const [index, code] of q.overlayCards.entries()) {
            const base = readCardWithReader(reader, code);
            if (!base) throw new Error(`Missing XYZ material in card database: ${code}`);
            cards.push({
              code, location: locationName(zone), sequence: q.sequence ?? 0, controller,
              position: 'faceup', overlay_sequence: index, attribute: pickBit(base.attribute, ATTR),
              race: pickBit(base.race, RACE), level: (base.level ?? 0) & 255, counter: 0,
              negated: false, attack: base.attack ?? 0, defense: base.defense ?? 0, types: typeNames(base.type),
            });
            if (cards.length === 160) break outer;
          }
        }
        const prefix = ({1:'',2:'h',4:'m',8:'s',16:'g',32:'r',64:'x'} as Record<number,string>)[zone];
        const hidden = pi === 1 && (privateZone || (!!((q.position ?? 0) & 10) &&
          !revealed.includes('o'+prefix+((q.sequence ?? 0)+1))));
        if (hidden) {
          cards.push({code:0,location:locationName(zone),sequence:q.sequence ?? 0,controller,
            position:privateZone?'none':positionName(q.position),overlay_sequence:-1,
            attribute:'none',race:'none',level:0,counter:0,negated:false,attack:0,defense:0,types:[]});
          if (cards.length === 160) break outer;
          continue;
        }
        const prince = zone === LOC.MZONE && !!((q.position ?? 0) & 5) && q.alias === 19144623;
        // The native tensor keeps the printed race even when Writer changes it.
        const printed = prince && q.code ? readCardWithReader(await room.getCardReader(), q.code) : undefined;
        cards.push({
          code: q.code ?? 0,
          ...(prince ? { public_prince_name: true } : {}),
          location: locationName(q.location ?? loc),
          sequence: q.sequence ?? 0,
          controller,
          // pos 为 0 时（如手牌）用 none；overlay 归一化交给 Python
          position: positionName(q.position),
          overlay_sequence: -1,
          attribute: pickBit(q.attribute, ATTR),
          race: pickBit(printed?.race ?? q.race, RACE),
          // models.py: rank/link 也算 level
          level: q.rank || q.link || q.level || 0,
          counter: q.counters?.[0]?.count ?? 0,
          negated: !!((q.status ?? 0) & (STATUS_DISABLED | STATUS_FORBIDDEN)),
          attack: q.attack ?? 0,
          // Link 怪兽规则上没有守备力，cdb 的 def 字段存的是连接箭头掩码。
          // 与 ygoenv/ygopro.h 的 `if (link_marker > 0) defense_ = link_marker`
          // 保持一致，让模型能看到箭头组合。
          defense: (q.linkMarker ?? 0) > 0 ? q.linkMarker : (q.defense ?? 0),
          types: typeNames(q.type),
        });
        if (cards.length === 160) break outer;
      }
    }
  }

  // LP 走 public 的 getCurrentFieldInfo（内部 queryFieldInfo）
  let myLp = 8000;
  let opLp = 8000;
  const info = await room.getCurrentFieldInfo();
  if (info && info.length >= 2) {
    myLp = info[myDuelPos]?.lp ?? 8000;
    opLp = info[1 - myDuelPos]?.lp ?? 8000;
  }

  return {
    cards,
    opponentSpellCounters: Math.min(opponentSpellCounters, 3),
    myLp,
    opLp,
    turn: room.turnCount || 1,
    phase: phaseName(room.phase),
    // OCG 里先手恒为 duelPos 0
    isFirst: myDuelPos === 0,
    isMyTurn: room.turnIngamePos === myDuelPos,
  };
}
