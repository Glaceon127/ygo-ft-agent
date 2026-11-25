// Track guaranteed identities independently of physical slots after a hand shuffle.
export class PublicHand {
  private pool: number[] = [];
  readonly slots = new Map<number, number>();
  shuffle() {
    this.pool.push(...Array.from(this.slots.values()).filter(Boolean));
    this.slots.clear();
  }
  names() {
    return [...this.pool, ...this.slots.values()]
      .filter(Boolean)
      .sort((a, b) => a - b);
  }
  private consume(code: number) {
    const i = this.pool.indexOf(code);
    if (i >= 0) this.pool.splice(i, 1);
  }
  reveal(slot: number, code: number) {
    if (!code) return;
    if (this.slots.has(slot)) {
      const old = this.slots.get(slot);
      if (!old || old === code) {
        this.slots.set(slot, code);
        return;
      }
      this.shuffle();
    }
    this.consume(code);
    this.slots.set(slot, code);
  }
  arrive(slot: number, code: number) {
    this.shuffle();
    if (code) this.pool.push(code);
    else this.slots.set(slot, 0);
  }
  leave(code: number) {
    this.shuffle();
    if (code) this.consume(code);
    else for (const name of new Set(this.pool)) this.consume(name);
  }
}
