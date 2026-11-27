import { buildActionMsg } from './action-builder';
import { FieldSnapshot } from './input-builder';
import { PublicHand } from './public-hand';
const effectTags: Record<string, number[]> = require('./effect_tags.json');

const quickEffects: Record<string, {quick: number[]; wait?: object}> = require('./quick_effects.json');
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
  private disableHint = false;
  private observedEffects = [new Map<number,number>(), new Map<number,number>()];
  private spentQuickNames = [0, 0];
  private spentQuickZones = [0, 0];
  // MSG_BECOME_TARGET follows MSG_CHAINING and precedes MSG_CHAINED.
  private announcedTargets = new Map<number, number>();
  private forgetTarget(c: any) {
    if (c.location !== 4) return;
    for (const [link, target] of this.announcedTargets) {
      if (target === c.controller * 8 + c.sequence) this.announcedTargets.set(link, -1);
    }
  }
  private clearQuickZone(c: any) {
    if (c.location !== 4 || c.sequence >= 7) return false;
    const used = !!(this.spentQuickZones[c.controller] & (1 << c.sequence));
    this.spentQuickZones[c.controller] &= ~(1 << c.sequence);
    return used;
  }
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
    if (n === 'YGOProMsgNewTurn') {
      for (const uses of this.observedEffects) uses.clear();
      this.spentQuickNames.fill(0);
      this.spentQuickZones.fill(0);
      this.disableHint = false;
    } else if (n === 'YGOProMsgPosChange' && (msg.currentPosition & 10)) {
      this.clearQuickZone(msg.card);
      this.forgetTarget(msg.card);
    }
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
      const spent = this.clearQuickZone(from);
      this.forgetTarget(from);
      this.clearQuickZone(to);
      if (spent && to.location === 4 && (to.position & 5))
        this.spentQuickZones[to.controller] |= 1 << to.sequence;
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
      this.forgetTarget(card1);
      this.forgetTarget(card2);
      const used1 = this.clearQuickZone(card1), used2 = this.clearQuickZone(card2);
      if (used1 && card2.location === 4) this.spentQuickZones[card2.controller] |= 1 << card2.sequence;
      if (used2 && card1.location === 4) this.spentQuickZones[card1.controller] |= 1 << card1.sequence;
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
      this.disableHint = msg.desc === 573;
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
      const origin = msg.chainCardLocation;
      const slot = msg.desc - msg.code * 16;
      if (origin && slot >= 0 && slot < 14) {
        const uses = this.observedEffects[origin.controller];
        uses.set(msg.code, (uses.get(msg.code) ?? 0) | (1 << slot));
      }
      if (origin?.location === 4) {
        if (msg.code === 55273560 && msg.desc === msg.code * 16) this.spentQuickNames[origin.controller] |= 1;
        if (msg.code === 26077387 && msg.desc === msg.code * 16) this.spentQuickNames[origin.controller] |= 2;
        if ([45819647,27552504].includes(msg.code) && msg.desc === msg.code * 16 + 1 &&
            msg.location === 4 && msg.controller === origin.controller)
          this.spentQuickZones[msg.controller] |= 1 << msg.sequence;
      }
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
    } else if (n === 'YGOProMsgBecomeTarget' && this.chainStack.length) {
      const link = this.chainStack[this.chainStack.length-1].link;
      const packed = msg.targets[0];
      const zone = (packed >>> 16) & 255;
      const single = !this.announcedTargets.has(link) && msg.targets.length === 1 &&
        ((packed >>> 8) & 255) === 4 && zone < 7;
      this.announcedTargets.set(link, single ? (packed & 255) * 8 + zone : -1);
    } else if (n === 'YGOProMsgChainEnd') {
      this.announcedTargets.clear();
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
      negation_context: [this.spentQuickNames[1-pos], this.spentQuickZones[1-pos],
        Number(this.disableHint && kind === 'YGOProMsgSelectCard')],
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
    const guide = !snap.isMyTurn && this.chainStack.length <= 8 &&
      this.chainStack[this.chainStack.length - 1]?.controller === 1-pos;
    const chain = msg.constructor.name === 'YGOProMsgSelectChain';
    const idle = msg.constructor.name === 'YGOProMsgSelectIdleCmd';
    if (!chain && !idle && msg.constructor.name !== 'YGOProMsgSelectEffectYn') return result;
    if (idle && !msg.activatableCards.some((c:any)=>c.code===10045474 && c.location===2)) return result;
    const enemies = snap.cards.filter(c => c.controller === 'opponent' && c.location === 'mzone' &&
      c.overlay_sequence === -1 && c.position.startsWith('faceup'));
    if (enemies.length < 2) return result;
    const top = this.chainStack[this.chainStack.length - 1];
    const activeQuick = top?.controller === 1-pos && top.location === 4 &&
      quickEffects[top.code]?.quick.includes(top.desc - top.code * 16 + 1);
    const ownRemoval = top?.controller === pos && !top.negated &&
      [83555666,4178474,79759861,12580477,53129443].includes(top.code) &&
      snap.cards.some(c => c.controller === 'opponent' && c.location === 'mzone' && c.overlay_sequence >= 0);
    const quick = (idle && enemies.some(c=>quickEffects[c.code]?.quick.length)) || ownRemoval || activeQuick || enemies.some(c =>
      [55273560,26077387,45819647,27552504].includes(c.code) || !!quickEffects[c.code]?.wait);
    const sources = enemies.filter(c => this.chainStack.some(e => e.controller === 1-pos &&
      e.location === 4 && e.code === c.code && e.sequence === c.sequence))
      .map(({code,sequence}) => ({code,sequence}));
    if (!quick && !(guide && sources.length)) return result;
    for (const [i,c] of (idle ? msg.activatableCards : chain ? msg.chains : [msg]).entries()) {
      if (idle && (c.code!==10045474 || c.location!==2)) continue;
      const dual = (c.code === 56725612 && c.location === 8) || (c.code === 78021082 && c.location === 4);
      if (c.code !== 97268402 && c.code !== 10045474 &&
          !(dual && (c.desc === c.code * 16 + 1 || (!chain && c.desc === 221)))) continue;
      const response = idle ? (i << 16) + 5 : chain ? i : 1;
      if (guide && sources.length) result[response] = await query({player:pos, response, sources});
      if (quick) {
        // Probe the real legal target menu on a disposable engine snapshot.
        // 128 marks a complete answer, bits 0..6 are legal monster zones.
        let zones = 128;
        for (const {code,sequence} of enemies) {
          const verdict = await query({player:pos, response, sources:[{code,sequence}]});
          if (!verdict) { zones = 0; break; }
          if (verdict === 2) zones |= 1 << sequence;
        }
        result['zones_' + response] = zones;
      }
    }
    return result;
  }

  input(snap: FieldSnapshot, msg: any, pos: number, selected: number[], preview: Record<string,number> = {}) {
    const action = buildActionMsg(msg, pos, selected);
    if (!action) return undefined;
    const context = this.context(pos, msg.constructor.name);
    const uses = Array<number>(14).fill(0);
    const targetZones = Array<number>(8).fill(0);
    for (const entry of this.chainStack) {
      const target = this.announcedTargets.get(entry.link);
      if (entry.controller === pos && !entry.negated && entry.link > 0 && entry.link <= 8 &&
          target !== undefined && target >= 0 && (target >> 3) === 1-pos)
        targetZones[entry.link-1] = 1 << (target & 7);
    }
    for (const c of snap.cards) {
      if (c.controller !== 'opponent' || c.location !== 'mzone' ||
          c.overlay_sequence !== -1 || !c.position.startsWith('faceup')) continue;
      const bits = this.observedEffects[1-pos].get(c.code) ?? 0;
      uses[2*c.sequence] = bits & 255;
      uses[2*c.sequence + 1] = bits >> 8;
    }
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
          negation_context: [...context.negation_context, snap.opponentSpellCounters ?? 0, ...uses, ...targetZones,
            Number(msg.constructor.name === 'YGOProMsgSelectChain' && ((msg.hint0 | msg.hint1) & 0x4) !== 0)],
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
