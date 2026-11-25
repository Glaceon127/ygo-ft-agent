// ygoinf HTTP 客户端
// 对照 ygoinf/server.py:
//   POST /v0/duels                  -> { duelId, index }
//   POST /v0/duels/{id}/predict     -> { predict_results: { action_preds[], win_rate }, index }
// 对照 ygoinf/models.py:
//   ActionPredict { prob: float, response: int, can_finish: bool }
export interface PredictResult {
  response: number;
  prob: number;
  /** select_sum 候选：该卡单独即可补完组合，调用方应立即收手发响应 */
  canFinish?: boolean;
}

export interface RankedPredict extends PredictResult {
  /** 在 action_preds 里的原始下标，用于回写 prev_action_idx */
  at: number;
}

interface RawPred {
  response: number;
  prob: number;
  can_finish?: boolean;
}

export class ModelClient {
  private duelId?: string;
  private index = 0;
  private prevActionIdx = 0;
  private busy = false;

  constructor(
    private baseUrl: string,
    private logger: { info: (o: unknown, m: string) => void; warn: (o: unknown, m: string) => void },
  ) {}

  /** 每局结束重置会话，避免 rstate / 历史动作跨局累积 */
  reset(): void {
    this.duelId = undefined;
    this.index = 0;
    this.prevActionIdx = 0;
  }

  async ensureDuel(): Promise<boolean> {
    if (this.duelId) return true;
    try {
      const res = await fetch(`${this.baseUrl}/v0/duels`, { method: 'POST' });
      if (!res.ok) throw new Error(`create ${res.status}`);
      const j = (await res.json()) as { duelId: string; index?: number };
      this.duelId = j.duelId;
      // index 缺失时必须兜底，否则 undefined 会被 JSON.stringify 丢掉，
      // 导致后续所有 predict 都 422（body.index Field required）。
      this.index = typeof j.index === 'number' ? j.index : 0;
      this.logger.info({ duelId: this.duelId, index: this.index }, 'ygoinf duel created');
      return true;
    } catch (e) {
      this.logger.warn(
        { error: e instanceof Error ? e.message : String(e) },
        'ygoinf create failed',
      );
      return false;
    }
  }

  /** 单次 HTTP 预测，返回原始 action_preds（未排序） */
  async _req(input: unknown): Promise<RawPred[] | undefined> {
    if (!this.duelId) return undefined;
    // server.py 内部维护单条 duel 的 rstate，不能并发
    if (this.busy) {
      this.logger.warn({}, 'predict skipped: busy');
      return undefined;
    }
    this.busy = true;
    try {
      const res = await fetch(`${this.baseUrl}/v0/duels/${this.duelId}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          input,
          prev_action_idx: this.prevActionIdx,
          index: this.index,
        }),
      });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(`${res.status} ${text.slice(0, 200)}`);
      }
      const j = (await res.json()) as {
        predict_results?: { action_preds?: RawPred[] };
        index?: number;
        error?: string;
      };
      const preds = j.predict_results?.action_preds;
      if (j.error || !preds?.some(p => p.prob >= 0) || j.index !== this.index + 1) {
        throw new Error(j.error || 'Invalid prediction or session index');
      }
      this.index = j.index;
      return preds;
    } catch (e) {
      this.logger.warn(
        { error: e instanceof Error ? e.message : String(e) },
        'predict failed',
      );
      // 一次失败会让服务端 duel 状态永久污损
      // （features.py: transform_select_idx 抛 ValueError，之后全 500）。
      // 这里丢弃 duelId，下次自动重建，避免错误累积。
      this.duelId = undefined;
      this.index = 0;
      this.prevActionIdx = 0;
      return undefined;
    } finally {
      this.busy = false;
    }
  }

  /** 返回模型选中的 response 整数值 */
  async predict(input: unknown): Promise<PredictResult | undefined> {
    const preds = await this._req(input);
    if (!preds) return undefined;
    // features.py::predict 里 state.update_history_actions(prev_action_idx)
    // 要求传「上一步选中的动作在 action_preds 里的下标」，
    // 传 0 会让 LSTM 的历史动作输入恒为第 0 个。
    let bestI = 0;
    for (let i = 1; i < preds.length; i += 1) {
      if (preds[i].prob > preds[bestI].prob) bestI = i;
    }
    this.prevActionIdx = bestI;
    // can_finish 只在 select_sum 候选里有意义（features.py: 该卡单独即可补完组合），
    // 插件靠它判断「现在必须收手发响应」，与训练侧组合耗尽即结束对齐。
    return {
      response: preds[bestI].response,
      prob: preds[bestI].prob,
      canFinish: !!preds[bestI].can_finish,
    };
  }

  /**
   * 返回全部候选，按 prob 降序，带原始下标 at（用于回写 prev_action_idx）。
   * 多选场景下插件需要「跳过 -1 占位/已选项再取下一个」，不能只要 top1。
   */
  async predictRanked(input: unknown): Promise<RankedPredict[] | undefined> {
    const preds = await this._req(input);
    if (!preds) return undefined;
    return preds
      .map((p, i) => ({
        response: p.response,
        prob: p.prob,
        at: i,
        canFinish: !!p.can_finish,
      }))
      .sort((a, b) => b.prob - a.prob);
  }

  /** 实际采纳了 at 这一项，历史动作要按它记，不能按 top1 记 */
  commit(at: number): void {
    this.prevActionIdx = at;
  }
}
