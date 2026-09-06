# プレフィックスキャッシュがエージェント負荷で一度も効かない — 問題と改善案

対象は FreeToken の `/v1/messages` 経路と hybrid（GDN）プレフィックスキャッシュ。
すべてこの箱（RTX 4090 24GB / i9-14900KF / 61GB RAM）での実測。

**第2版（2026-09-07）。** 初版に対する査読で問題2の機構の説明が浅いと指摘され、書き直した。
査読で解決した項目は「査読で解決した点」に移した。

---

## 要約

Claude Code の実トラフィックで、プレフィックスキャッシュが**会話のターンに対して一度も効いていない**。
原因は2つあり、独立している。

1. **主因（`anthropic_api.py`）** — 会話の途中に置かれた `role:"system"` メッセージを
   先頭の system ブロックに巻き上げて連結している。クライアントはターンごとに新しい
   `<system-reminder>` を送るので、**毎ターン約19トークンがプロンプトの先頭側に増える**。
   その後ろの 80k〜130k は原理的に一致しない。
2. **従因（`hybrid_radix_cache.py` / `scheduler.py`）** — 一致した分すら捨てている。
   GDN スナップショットは insert の末尾にしか置かれず、再利用したい位置には1つも無い。
   **これは実装漏れではなく、overlap スケジューリング下の二重解放を避けるための意図的な抑制である。**

主因だけを直すと、実 body 58 組で**改善 41 / 悪化 0 / 変化なし 17**。
一致長は最大で **21,623 → 105,057 トークン（全体の 99.4%）** になる（オフライン実測）。

**推奨: まず案A を実機で確認する。案C は独立した、より重い変更として別に扱う。**

---

## 計測環境と窓

```
ft serve --model-path ornith-ai/Ornith-1.5-35B-A3B-NVFP4
         --kv-reserve-tokens 262144 --moe-backend offload
         --max-running-requests 16 --port 1919
```

* 窓: 2026-09-07 01:01:38 – 01:19:45（約18分）、Claude Code の実セッション1本
* リクエスト 61 件（本文 100〜400 KB）
* 計装: `FREETOKEN_PREFIX_DEBUG=1`（match / insert / walk を1行ずつ）、
  `FREETOKEN_WIRE_BODY_DIR`（リクエスト本文を1件1ファイルで保存）— 枝 `diag/prefix-cache`

### 観測（Prefill バッチ 376 件、重複しない実計算量）

```
cached-token > 0 の行 :  18 件   計算     3,904 トークン   再利用 1,886,642
cached-token = 0 の行 : 358 件   計算 2,822,665 トークン   再利用         0
```

**当たるときはほぼ全部当たり、外れるときは全部外れる。中間が無い。**
当たっている 18 件はすべて補助呼び出し（要約・タイトル生成など、同じ前置きを丸ごと共有するもの）。
**会話のターンは 358 件すべて外れている。**

---

## 問題1（主因）: 会話途中の system メッセージを先頭へ巻き上げる

### 現象

`walk` 行（トークン一致長、スナップショット切り捨ての**前**）:

```
tok_match=21547   tok_match=21566   tok_match=21585   tok_match=21604 ...
```

リクエストは 130k トークン超まで育つのに、一致長は **21.5k で貼り付き、毎ターン +19 ずつしか伸びない**。

### 機構

`python/freetoken/server/anthropic_api.py:203-218, 287-290`

```python
# Collect all system content (top-level `system` + any system-role messages
# Claude Code interleaves in the array) and emit ONE system message at the
# front: strict chat templates (e.g. Qwen3.5) require system at the beginning.
system_texts: list[str] = []
...
    for msg in req.messages:
        if msg.role == "system":
            system_texts.append(_content_text(msg.content))
            continue
...
system_text = "\n\n".join(t for t in system_texts if t)
if system_text:
    messages.append({"role": "system", "content": system_text})
messages.extend(other)
```

最小再現（`convert_anthropic_prompt` の戻り値）:

```
入力 messages: user U1 / assistant A1 / system R1 / user U2 / assistant A2 / system R2 / user U3

出力 messages:
  system     BASE-SYSTEM\n\nR1\n\nR2      ← 巻き上げて連結
  user       U1
  assistant  A1
  user       U2
  assistant  A2
  user       U3
```

クライアントは実際にこう送ってくる（実 body より。値はターンごとに変わる）:

```
req-00002: .messages[4].content[0].text  = <total_tokens>14974723 tokens left</total_tokens>
req-00003: .messages[7].content[0].text  = 14973675      ← 1つ増える
req-00004: .messages[10].content[0].text = 14972594      ← また増える
```

レンダリング後の実際の分岐点（`convert_anthropic_prompt` + chat template を通した実測）:

```
A: ...run concurrently.\n\n<total_tokens>14974723 tokens left</total_tokens> ▲ <|im_end|><|im_start|>user
B: ...run concurrently.\n\n<total_tokens>14974723 tokens left</total_tokens> ▲ \n\n<total_tokens>14973675 ...
```

**巻き上げには理由がある。** Ornith の `tokenizer_config.json` のテンプレートは途中の system を拒否する:

```
TemplateError: System message must be at the beginning.
```

つまり「巻き上げをやめる」だけでは直らない。**在置きの表現を選ぶ必要がある。**

### 影響（実 body 58 組、GPU 不使用のオフライン実測）

途中の system を「直後の user turn にマージ」して同じ変換を通したときの**トークン一致長**。

```
58 組:  改善 41  /  悪化 0  /  変化なし 17
```

「変化なし」17 組はすべて、reminder が挟まっていない組（＝既に当たっている補助呼び出し）。
**在置きにして悪化した組は無い。**

セッションの進行につれ、現状の一致率は下がり続ける（一致長が 21.5k で凍る一方、プロンプトが育つため）:

| 連続する2件 | 現状 | 在置き | B の総トークン |
|---|---:|---:|---:|
| req-00001 → 00002 | 20,557 (78.7%) | 23,232 (88.9%) | 26,142 |
| req-00008 → 00009 | 21,245 (57.9%) | 36,529 (99.5%) | 36,701 |
| req-00019 → 00020 | 21,490 (22.4%) | 86,223 (89.9%) | 95,880 |
| req-00031 → 00032 | 21,623 (20.5%) | **105,057 (99.4%)** | 105,722 |
| req-00054 → 00055 | 21,851 (16.1%) | **131,940 (97.4%)** | 135,448 |
| req-00020 → 00021 | 95,816 (99.8%) | 95,850 (99.8%) | 96,033 |
| req-00035 → 00036 | 106,470 (99.8%) | 106,540 (99.8%) | 106,733 |

（最後の2行が「変化なし」の組。悪化していないことの例として載せた。）

**根拠の所在**（初版への査読指摘を受けて、揮発しない場所へ移した）:

* 全 58 組: `.claude/REVIEW-prefix-cache-evidence.json`（本文書と同じコミットに入っている。
  長さと本文のハッシュのみ。**会話の中身は含まない**）
* 再導出: `freetoken-systest/tools/prefix-hoist-compare.py`（GPU 不要、トークナイザのみ）
* 一次出力: `freetoken-systest/results/20260907T013123-diag_prefix-cache-hoist-compare.json`

---

## 問題2（従因）: GDN スナップショットが insert の末尾にしか置かれない

### 一致した分を捨てる機構

`python/freetoken/kvcache/hybrid_radix_cache.py:76-88` —
トークン一致のあと、**生きたスナップショットを持つ最深ノードまで遡って切り捨てる**。
経路上に1つも無ければ `cached_len=0` を返し、**一致していた KV を全部捨てる**。

```python
cur, end_len = node, self._path_len(node)
while not cur.is_root():
    if cur.mamba_value is not None:
        return HybridMatch(self._collect_kv(cur), end_len, cur.mamba_value, cur)
    end_len -= cur.length
    cur = cur.parent
return HybridMatch(self.empty, 0, None, self.root)   # 全部捨てる
```

### なぜ再利用したい位置にスナップショットが無いのか

**実装漏れではない。意図的に抑制されている。**
`python/freetoken/scheduler/scheduler.py:320-323`、`_process_last_data` の中:

```python
if isinstance(req, ChunkedReq):
    # Don't cache intermediate chunks; the full prompt is cached once when the
    # final chunk is processed. Caching here snapshots a handle the next chunk
    # already copied (overlap), so cache_req double-frees the prior chunk.
```

overlap スケジューリングでは次のチャンクの forward が既に走っており、その handle をコピー済みである。
途中チャンクで `cache_req` すると、コピー済みの handle をスナップショットしてしまい、
前のチャンクを二重解放する。**したがって「途中でも donate する」は、
まず overlap 下の handle の所有権を解いてからでないと入れられない。**

その結果、`linear.py:113-128` が forward ごとに `mamba_last_track_seqlen` を上書きしても
（ping-pong は2スロット）、donate まで生き残るのは最後の1個だけになる。
25,061 トークンのプロンプトは 8192 で4回に刻まれて計算されるのに、置かれるスナップショットは
25,024 = 391×64 の1個。

```
置かれた位置  25024 25277 26112 26325 27136 ... 38080 ... 104128
必要な位置    20557 21169 21188 21207 21226 21245 ... 21851
                                                     ↑ 重なりゼロ
```

**資源は制約ではない。** GDN スロットはセッション中ずっと最大 **8/96**、KV 占有は最大 **0.55**。
96 は `max_running_requests 16 × 6`（査読者の確認。`--max-running-requests 4` では 24 になる）。
追い出し圧力ではない。

---

## 改善案

### 案A — 会話途中の system を、直後の user turn にマージして在置き（問題1）

* **効果（実測）**: 58 組で改善 41 / 悪化 0。最大 21,623 → 105,057 / 105,722（99.4%）
* テンプレート適合: 確認済み（レンダリング成功）
* クライアント自身が最初の reminder を user ブロックの先頭に置いているので、
  **モデルが見る構造が一貫する**（査読者も B より筋がいいと評価）
* 巻き上げが必要なテンプレートでも動く（system は先頭のまま）
* **副作用**: role が `system` から `user` に変わるので、**reminder の効き方は変わりうる**。
  出力品質で1回見ておく価値がある

### 案B — 会話途中の system を、独立した user turn として在置き（問題1）

* テンプレート適合: 確認済み（連続する user turn を受ける）
* turn 数が増える。案A より元の構造に忠実
* 一致長への効果は案A と同等と見込むが、**未測定**

### 案C — チャンク prefill の途中でもスナップショットを donate する（問題2）

* **「機構があるので呼ぶだけ」ではない。** 上記のとおり `_process_last_data` は
  overlap 下の二重解放を避けるために意図して飛ばしている。
  **overlap スケジューリングにおける handle の所有権を先に解く必要がある**変更であり、
  見積もりは案A/B より一段重い
* 資源には余裕がある（8/96）
* コスト: donate ごとに約1MBのスロットコピー（`_clone_slot_for_tree`）。
  8192 刻みなら1リクエストあたり donate が1回→3〜4回
* 参考: `_clone_slot_for_tree` は所有権共有で実際に壊れた事例
  （並行負荷で約10%の hit が汚染）を記録している。この領域は既に一度事故っている

### 案の関係

案A または案B を入れると**分岐点が会話の末尾に移る**。末尾は現状で唯一スナップショットが
置かれる場所なので、**案C 無しでも効く見込み**。

**ただしこれには条件がある。** 次ターンの前置きが「前の prompt + 前の assistant 応答 + 新ターン」
であり、前の要求の終了時スナップショットがその経路上にあることに依存している。
つまり**クライアントが assistant 応答を逐語で送り返すこと**が前提である。
Claude Code はそうしているが、**応答を正規化して送るクライアントでは分岐点がスナップショットの
手前に戻り、案C が無いと当たらない**（査読者の指摘）。

**推奨: まず案A を実機で確認。案C は上記の条件に該当するクライアントのための別件として扱う。**

---

## 査読で解決した点（初版の未判断項目）

* **`_process_last_data` が ChunkedReq を飛ばす理由** — overlap 下の二重解放を避けるため。
  コード内のコメントに明記されている。初版の「呼ばれていないだけ」「機構は既にある」は誤り。
  問題2 と案C を書き直した
* **同一要求の再照合（`tok_match=21680` が 2,616 回連続）** — 正常動作。
  `prefill.py:76` の `_try_allocate_one` が `match_req` を呼び、
  それが `try_add_one`（`prefill.py:261`）経由で**周回ごとに待機中の全要求に対して**走る。
  受け入れられない要求が1本あるだけで、100k トークンの木の歩行が周回ごとに繰り返される。
  **正常だが無駄で、待ち行列が深いとスケジューラのループ時間を実際に食う**
  （査読者の指摘。FINDINGS #1 の 236 本が溜まる状況が該当）。
  木が変わっていない間は照合結果を要求側に持たせれば消える。**別件。**
* **GDN スロット 96 の由来** — `max_running_requests 16 × 6`。資源が制約でないという判断は妥当

---

## 未測定・未判断（引き続き査読でつぶしてほしい点）

1. **端から端までの速度を測っていない。** 上の表はトークン一致長であって時間ではない。
2. **案A で出力品質が保たれるかを評価していない。** role が system から user に変わるため、
   reminder の効き方は変わりうる。**測るというより決める話**だが、1回は見ておくべき。
3. **巻き上げは無条件。** 途中の system を許すテンプレートでは巻き上げ自体が不要ではないか。
   テンプレート能力で分岐すべきか、常に在置きにすべきか。
4. 以前記録した「起動直後は同じ前置きを4連続で投げても全部 prefill する」は、
   今回のデータでは説明できていない。別現象の可能性がある。**相関のまま置いている。**

---

## 再現手順

計装は枝 `diag/prefix-cache`（Python のみ。`.so` の再ビルド不要。env が無ければ経路は変わらない）。

```
FREETOKEN_PREFIX_DEBUG=1 \
FREETOKEN_WIRE_LOG=<path> \
FREETOKEN_WIRE_BODY_DIR=<dir> \
ft serve --model-path ... --kv-reserve-tokens 262144 --moe-backend offload --port 1919
```

* `prefix-cache: walk tok_match=<切り捨て前> -> snap_trunc=<後>` で、
  「プロンプトが違う」のか「一致したのに捨てた」のかが1行で分かる
* リクエスト本文の分岐点: `freetoken-systest/tools/prefix-divergence.py <dir>`
* 巻き上げ有無の一致長比較: `freetoken-systest/tools/prefix-hoist-compare.py <dir> --snapshot <ckpt>`
  （GPU 不要）

**注意**: `FREETOKEN_WIRE_BODY_DIR` に落ちるのは**会話の全文**である。保存先の扱いに注意すること。
本文書の根拠は、本文を含まない `.claude/REVIEW-prefix-cache-evidence.json` に落としてある。

---

## 副次的な観測（本題ではないが、同じ計装で見えた）

* **expert ロード中に `/v1/models` が 200 を返す。** uvicorn の
  "Application startup complete" はモデルロードの**前**に出るため、readiness に使えない。
  **査読者が同日、独立に同じ現象を踏んでいる**（生成が通るまで待つ形に直したとのこと）。
* **バックエンド worker が死ぬとフロントエンドが SIGTERM を無視した。**
  `_exit_after_backend_death`（`api_server.py:90`）が自分に SIGTERM を送り、
  uvicorn が背景タスクの完了を待って止まる型。"Waiting for background tasks to complete" は
  uvicorn 自身の文言。15秒待っても落ちず SIGKILL が必要だった。
  SIGKILL で pidfile が残るのは**今日直した経路の外**（SIGKILL は何も走らせない）で、
  そこは stale 引き継ぎが受け持つ設計。**1回のみ観測。再現できたら別件として起こす。**

---

本文書の作成には Claude Opus 5 を用いた。すべての数値はこの箱での実測で、
`.claude/REVIEW-prefix-cache-evidence.json` と `freetoken-systest/results/` に辿れる。
