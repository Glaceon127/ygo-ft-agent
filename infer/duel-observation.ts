import { buildActionMsg } from './action-builder';
import { FieldSnapshot } from './input-builder';
import { PublicHand } from './public-hand';
const effectTags: Record<string, number[]> = require('./effect_tags.json');

export class DuelObservation {
  chainStack: Array<{
    controller: number;
    code: number;
    location: number;
    sequence: number;
    desc: number;
    link: number;
    negated: boolean;
  }> = [];
  hands = [new PublicHand(), new PublicHand()];
  attackSource?: { controller: number; sequence: number };
  targetHint = false;
  chainForced = false;
  speCount = 0;
  sawNegated = false;
  private instantFusionPending = false;
  private discardHint = false;
  private discards = new WeakSet<object>();
  private temporaryMonsters = [new Set<number>(), new Set<number>()];
  revealed: Array<{ controller: number; location: number; sequence: number }> =
    [];
  // Keep the activation zone in the chain. Slot 74 is outside every monster
  // zone and within the policy's sequence embedding; a replacement card
  // entering the vacated slot must not become the source of the effect.
  private movedSource(entry: (typeof this.chainStack)[number], from: any, to: any) {
    if (entry.location === 4 && from.location === 4 &&
        entry.controller === from.controller && entry.sequence === from.sequence) {
      entry.sequence = to.location === 4 && to.controller === from.controller ? to.sequence : 74;
      return true;
    }
    return false;
  }
  observe(msg: any) {
    const n = msg.constructor.name;
    if (n === 'YGOProMsgConfirmCards') {
      for (const c of msg.cards ?? [])
        if (c.location === 2 && msg.player === 1 - c.controller)
          this.hands[c.controller].reveal(c.sequence, c.code);
    } else if (n === 'YGOProMsgShuffleHand') {
      this.hands[msg.player].shuffle();
    } else if (n === 'YGOProMsgDraw') {
      for (const code of msg.cards ?? [])
        if (code >>> 31 !== 0)
          this.hands[msg.player].arrive(0, code & 0x7fffffff);
    } else if (n === 'YGOProMsgMove') {
      const from = msg.previous,
        to = msg.current;
      for (const entry of this.chainStack) this.movedSource(entry, from, to);
      const visible = (c: any) =>
        c.location === 16 || (c.location & (4 | 8 | 32 | 64) && c.position & 5);
      const code = visible(from) || visible(to) ? msg.code & 0x7fffffff : 0;
      if (from.location === 4) this.temporaryMonsters[from.controller].delete(from.sequence);
      if (this.instantFusionPending && from.location === 64 && to.location === 4) {
        this.temporaryMonsters[to.controller].add(to.sequence);
        this.instantFusionPending = false;
      }
      if (from.location === 2) this.hands[from.controller].leave(code);
      if (to.location === 2)
        this.hands[to.controller].arrive(to.sequence, code);
      if (
        from.location === 4 &&
        this.attackSource?.controller === from.controller &&
        this.attackSource?.sequence === from.sequence
      ) {
        this.attackSource = to.location === 4 && to.controller === from.controller ? to : undefined;
      }
    } else if (n === 'YGOProMsgSwap') {
      const { card1, card2 } = msg;
      for (const entry of this.chainStack) {
        if (!this.movedSource(entry, card1, card2)) this.movedSource(entry, card2, card1);
      }
      for (const card of [card1, card2]) {
        if (card.location === 2) this.hands[card.controller] = new PublicHand();
      }
      const source = this.attackSource;
      if (source) {
        for (const [from, to] of [
          [card1, card2],
          [card2, card1],
        ]) {
          if (
            from.location === 4 &&
            from.controller === source.controller &&
            from.sequence === source.sequence
          ) {
            this.attackSource = to.location === 4 && to.controller === from.controller ? to : undefined;
            break;
          }
        }
      }
    } else if (n === 'YGOProMsgHint' && msg.type === 3) {
      this.targetHint = msg.desc === 549;
      if (msg.desc === 501) this.discardHint = true;
    } else if (n === 'YGOProMsgSelectCard' && this.discardHint) {
      this.discards.add(msg);
      this.discardHint = false;
    } else if (n === 'YGOProMsgNewPhase') {
      this.targetHint = false;
      this.attackSource = undefined;
    }
    if (n === 'YGOProMsgChainSolved') this.revealed = [];
    if (n === 'YGOProMsgConfirmCards')
      this.revealed.push(
        ...(msg.cards ?? []).filter((c: any) => c.location !== 2),
      );
    if (n === 'YGOProMsgChaining') {
      if (msg.code === 1845204) this.instantFusionPending = true;
      if (msg.location === 2)
        this.hands[msg.controller].reveal(msg.sequence, msg.code);
      this.chainStack.push({
        controller: msg.chainCardLocation?.controller ?? msg.controller ?? 0,
        code: msg.code ?? 0,
        location: msg.chainCardLocation?.location ?? msg.location ?? 0,
        sequence: msg.chainCardLocation?.location === 4
          ? (msg.location === 4 && msg.controller === msg.chainCardLocation.controller ? msg.sequence : 74)
          : (msg.chainCardLocation?.sequence ?? msg.sequence ?? 0),
        link: msg.chainCount,
        desc: msg.desc >>> 0,
        negated: false,
      });
    } else if (n === 'YGOProMsgChainEnd') {
      this.chainStack = [];
      this.chainForced = false;
      this.sawNegated = false;
    } else if (
      n === 'YGOProMsgChainNegated' ||
      n === 'YGOProMsgChainDisabled'
    ) {
      this.sawNegated = true;
      const link = this.chainStack.find((e) => e.link === msg.chainCount);
      if (link) link.negated = true;
    } else if (n === 'YGOProMsgSelectChain') {
      this.speCount = msg.specialCount ?? 0;
      const ch = msg.chains ?? [];
      this.chainForced = ch.some((c: any) => (c?.forced ?? 0) !== 0);
    }
  }
  committed(msg: any, response: number) {
    if (msg.constructor.name === 'YGOProMsgSelectBattleCmd') {
      const card = msg.attackableCards[response >>> 16];
      this.attackSource = (response & 0xffff) === 1 && card ? card : undefined;
      this.targetHint = false;
    } else if (msg.constructor.name === 'YGOProMsgSelectCard') {
      this.targetHint = false;
    }
  }
  automaticDiscard(msg: object, phase: number) {
    return phase === 0x200 && this.discards.has(msg);
  }
  context(pos: number, kind: string) {
    return {
      revealed: this.revealed.map((c) => {
        const prefix = (
          {
            1: '',
            2: 'h',
            4: 'm',
            8: 's',
            16: 'g',
            32: 'r',
            64: 'x',
          } as Record<number, string>
        )[c.location];
        return (c.controller === pos ? '' : 'o') + prefix + (c.sequence + 1);
      }),
      public_hand: this.hands[1 - pos].names(),
      public_slots: Object.fromEntries(this.hands[1 - pos].slots),
      attack_target: this.targetHint && kind === 'YGOProMsgSelectCard',
      attacker:
        this.attackSource?.controller === pos
          ? 'm' + (this.attackSource.sequence + 1)
          : null,
      entries: this.chainStack.map((ce) => {
        const L = ce.location;
        const ch =
          L & 2
            ? 'h'
            : L & 4
              ? 'm'
              : L & 8
                ? 's'
                : L & 16
                  ? 'g'
                  : L & 32
                    ? 'r'
                    : L & 64
                      ? 'x'
                      : '';
        return {
          ...ce,
          opponent: ce.controller !== pos,
          spec: (ce.controller === pos ? '' : 'o') + ch + (ce.sequence + 1),
          on_field: (L & 12) !== 0,
        };
      }),
    };
  }

  async negationPreview(snap: FieldSnapshot, msg: any, pos: number,
    query: (offer: {player:number; response:number; sources:{code:number; sequence:number}[]}) => Promise<number>) {
    const result: Record<string,number> = {};
    if (snap.isMyTurn || this.chainStack.length > 8 ||
        this.chainStack[this.chainStack.length - 1]?.controller !== 1-pos) return result;
    const chain = msg.constructor.name === 'YGOProMsgSelectChain';
    if (!chain && msg.constructor.name !== 'YGOProMsgSelectEffectYn') return result;
    const enemies = snap.cards.filter(c => c.controller === 'opponent' && c.location === 'mzone' &&
      c.overlay_sequence === -1 && c.position.startsWith('faceup'));
    if (enemies.length < 2) return result;
    const sources = enemies.filter(c => this.chainStack.some(e => e.controller === 1-pos &&
      e.location === 4 && e.code === c.code && e.sequence === c.sequence))
      .map(({code,sequence}) => ({code,sequence}));
    if (!sources.length) return result;
    for (const [i,c] of (chain ? msg.chains : [msg]).entries()) {
      if (c.code !== 97268402 && c.code !== 10045474 &&
          !([56725612,78021082].includes(c.code) && c.desc === c.code * 16 + 1)) continue;
      const response = chain ? i : 1;
      result[response] = await query({player:pos, response, sources});
    }
    return result;
  }

  input(snap: FieldSnapshot, msg: any, pos: number, selected: number[], preview: Record<string,number> = {}) {
    const action = buildActionMsg(msg, pos, selected);
    if (!action) return undefined;
    const context = this.context(pos, msg.constructor.name);
    const myChain = this.chainStack.filter(
      (entry) => entry.controller === pos,
    ).length;
    const tags = this.chainStack.map(
      (entry) => (effectTags as Record<string, number[]>)[String(entry.code)],
    );
    return {
      global: {
        my_lp: snap.myLp,
        op_lp: snap.opLp,
        turn: snap.turn,
        phase: snap.phase,
        is_first: snap.isFirst,
        is_my_turn: snap.isMyTurn,
        chain_info: {
          ...context,
          negation_preview: preview,
          chain_count: Math.min(this.chainStack.length, 7),
          my_chain: Math.min(myChain, 7),
          op_chain: Math.min(this.chainStack.length - myChain, 7),
          last_chain_is_op:
            context.entries[context.entries.length - 1]?.opponent ?? false,
          has_negate: tags.some((tag) => tag?.[0]),
          has_destroy: tags.some((tag) => tag?.[1]),
          chain_forced: this.chainForced,
          has_negated: this.sawNegated,
          spe_count: Math.min(this.speCount, 7),
        },
      },
      cards: snap.cards.map(card => {
        const owner = card.controller === 'me' ? pos : 1-pos;
        if (card.location === 'mzone' && card.code && !card.negated &&
            this.temporaryMonsters[owner].has(card.sequence)) return {...card,temporary:true};
        return card;
      }),
      action_msg: { data: action },
    };
  }
}
