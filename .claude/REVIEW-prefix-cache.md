# プレフィックスキャッシュがエージェント負荷で一度も効かない — 問題と改善案

査読依頼。対象は FreeToken の `/v1/messages` 経路と hybrid（GDN）プレフィックスキャッシュ。
すべてこの箱（RTX 4090 24GB / i9-14900KF / 61GB RAM）での実測。数値はローカルのログに辿れる。

---

## 要約

Claude Code の実トラフィックで、プレフィックスキャッシュが**会話のターンに対して一度も効いていない**。
原因は2つあり、独立している。

1. **主因（`anthropic_api.py`）** — 会話の途中に置かれた `role:"system"` メッセージを
   先頭の system ブロックに巻き上げて連結している。クライアントはターンごとに新しい
   `<system-reminder>` を送るので、**毎ターン約19トークンがプロンプトの先頭側に増える**。
   その後ろの 80k〜100k は原理的に一致しない。
2. **従因（`hybrid_radix_cache.py` / `scheduler.py`）** — 一致した分すら捨てている。
   GDN スナップショットは insert の末尾にしか置かれず、再利用したい位置には1つも無い。

主因だけを直すと、実 body での一致長は **21,623 → 105,057 トークン（全体の 99.4%）** になる（オフライン実測）。

---

## 計測環境と窓

```
ft serve --model-path ornith-ai/Ornith-1.5-35B-A3B-NVFP4
         --kv-reserve-tokens 262144 --moe-backend offload
         --max-running-requests 16 --port 1919
```

* 窓: 2026-09-07 01:01:38 – 01:19:45（約18分）、Claude Code の実セッション1本
* リクエスト 61 件（うち 10KB 超が 57 件、本文は 100〜400 KB）
* 計装: `FREETOKEN_PREFIX_DEBUG=1`（match / insert / walk を1行ずつ）、
  `FREETOKEN_WIRE_BODY_DIR`（リクエスト本文を1件1ファイルで保存）

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

`FREETOKEN_PREFIX_DEBUG=1` の `walk` 行（トークン一致長、スナップショット切り捨ての**前**）:

```
tok_match=21547   tok_match=21566   tok_match=21585   tok_match=21604 ...
```

リクエストは 100k トークン超まで育つのに、一致長は **21.5k で貼り付き、毎ターン +19 ずつしか伸びない**。

### 機構

`python/freetoken/server/anthropic_api.py:203-218, 287-289`

```python
# Collect all system content (top-level `system` + any system-role messages
# Claude Code interleaves in the array) and emit ONE system message at the
# front: strict chat templates (e.g. Qwen3.5) require system at the beginning.
system_texts: list[str] = []
...
    if msg.role == "system":
        system_texts.append(_content_text(msg.content))
...
system_text = "\n\n".join(t for t in system_texts if t)
if system_text:
    messages.append({"role": "system", "content": system_text})
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

クライアントは実際にこう送ってくる（実 body より）:

```
req-00002: .messages[4].content[0].text  = <total_tokens>14974723 tokens left</total_tokens>
req-00003: .messages[4].content          = 14974723
           .messages[7].content[0].text  = 14973675      ← 1つ増える
req-00004: .messages[10].content[0].text = 14972594      ← また増える
```

レンダリング後の実際の分岐点（`convert_anthropic_prompt` + chat template を通した実測）:

```
A: ...run concurrently.\n\n<total_tokens>14974723 tokens left</total_tokens> ▲ <|im_end|><|im_start|>user
B: ...run concurrently.\n\n<total_tokens>14974723 tokens left</total_tokens> ▲ \n\n<total_tokens>14973675 ...
```

**巻き上げには理由がある。** Ornith のテンプレートは途中の system を拒否する:

```
TemplateError: System message must be at the beginning.
```

つまり「巻き上げをやめる」だけでは直らない。**在置きの表現を選ぶ必要がある。**

### 影響（MT の実 body、GPU 不使用のオフライン実測）

途中の system を「直後の user turn にマージ」して同じ変換を通したときの**トークン一致長**:

| 連続する2件 | 現状 | 在置き | B の総トークン |
|---|---:|---:|---:|
| req-00002 → 00003 | 21,169 | 25,522 | 27,160 |
| req-00003 → 00004 | 21,188 | 27,133 | 27,724 |
| req-00004 → 00005 | 27,704 | 27,694 | 27,858 |
| req-00011 → 00012 | 21,302 | 37,000 | 37,596 |
| req-00020 → 00021 | 95,816 | 95,850 | 96,033 |
| **req-00031 → 00032** | **21,623** | **105,057** | **105,722（99.4%）** |

21.5k で頭打ちだった組がプロンプトのほぼ全長まで伸びる。
既に伸びている組（reminder が挟まっていない組）は**在置きにしても悪化しない**。

---

## 問題2（従因）: GDN スナップショットが insert の末尾にしか置かれない

### 機構

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

`python/freetoken/scheduler/scheduler.py:405` — **チャンク prefill の途中では donate しない**:

```python
elif batch.is_prefill and req.table_idx != -1:
    # for prefill, non-chunk req, cache the prefix.
    self.cache_manager.cache_req(req, finished=False)
```

`python/freetoken/attention/linear.py:113-128` — `mamba_last_track_seqlen` は forward ごとに上書きされ、
ping-pong は2スロットしかない。結果、**1リクエストにつきスナップショットは末尾の1個だけ**になる。

### 観測

25,061 トークンのプロンプトは 8192 で4回に刻まれて計算されるのに、置かれるスナップショットは 25,024 の1個。

```
置かれた位置  25024 25277 26112 26325 27136 ... 38080 ... 104128
必要な位置    20557 21169 21188 21207 21226 21245 ... 21585
                                                     ↑ 重なりゼロ
```

**資源は制約ではない。** セッション中ずっと GDN スロットは最大 **8/96**（88個が遊休）、
KV 占有は最大 **0.55**。追い出し圧力ではない。

---

## 改善案

### 案A — 会話途中の system を、直後の user turn にマージして在置き（問題1）

* **効果（実測）**: 一致長 21,623 → 105,057 / 105,722（99.4%）
* テンプレート適合: 確認済み（レンダリング成功）
* クライアント自身が最初の reminder を user ブロックの先頭に置いているので、形は一貫する
* 巻き上げが必要なテンプレートでも動く（system は先頭のまま）

### 案B — 会話途中の system を、独立した user turn として在置き（問題1）

* テンプレート適合: 確認済み（連続する user turn を受ける）
* turn 数が増える。案A より元の構造に忠実
* 一致長への効果は案A と同等と見込むが、**未測定**

### 案C — チャンク prefill の途中でもスナップショットを donate する（問題2）

* **機構は既にある。** `_cache_req_hybrid` の "Prefill chunk commit" 経路が
  `mamba_last_track_seqlen` で donate する処理を持っている。呼ばれていないだけ
* 資源に余裕あり（8/96）
* コスト: donate ごとに約1MBのスロットコピー（`_clone_slot_for_tree` のコメントによる）。
  8192 刻みなら1リクエストあたり donate が1回→3〜4回に増える
* **リスク: `scheduler.py:405` の "non-chunk req" 制限が何を守っているのか未確認。**
  `_clone_slot_for_tree` は所有権共有で実際に壊れた事例（並行負荷で約10%の hit が汚染）を記録している

### 案の関係

案A または案B を入れると**分岐点が会話の末尾に移る**。末尾は現状で唯一スナップショットが
置かれる場所なので、**案C 無しでも効く見込み**（合成形が同じ形で当たった、という間接証拠）。
案C は独立に価値がある —— 履歴の途中が変わる他のクライアントや、案A/B を適用できない場合に効く。

**推奨: まず案A を実機で確認。効けば案C は別件として扱う。**

---

## 未測定・未判断（査読でつぶしてほしい点）

1. **端から端までの速度を測っていない。** 上の表はトークン一致長であって時間ではない。
2. **案A で出力品質が保たれるかを評価していない。** reminder の位置が変わればモデルの反応は変わりうる。
   案A と案B のどちらが正しいかは、測るというより決める話だと考えている。
3. **`scheduler.py:405` の "non-chunk req" 制限の理由が分かっていない。**
   チャンク途中の donate が安全かどうか、意図を知る人の判断が要る。
4. **巻き上げは無条件。** 途中の system を許すテンプレートでは巻き上げ自体が不要ではないか。
   テンプレート能力で分岐すべきか、常に在置きにすべきか。
5. **同一リクエストが再照合されている。** `tok_match=21680 -> snap_trunc=0` が **2,616 回連続**で出る。
   スケジューラの周回ごとに `match_req` が走っているように見える。正常か、無駄か、未判断。
6. 以前記録した「起動直後は同じ前置きを4連続で投げても全部 prefill する」は、今回のデータでは説明できていない。
   別現象の可能性がある。**相関のまま置いている。**

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
* リクエスト本文の分岐点は `freetoken-systest/tools/prefix-divergence.py <dir>`
* 巻き上げ有無の一致長比較はオフライン（GPU 不要。`convert_anthropic_prompt` +
  `apply_chat_template` + トークナイザのみ）

---

## 副次的な観測（本題ではないが、同じ計装で見えた）

* **expert ロード中に `/v1/models` が 200 を返す。** uvicorn の
  "Application startup complete" はモデルロードの**前**に出るため、readiness に使えない。
* **バックエンド worker が死ぬとフロントエンドが SIGTERM を無視する。**
  "Backend worker is gone and cannot be restarted; stopping the API server" を出したあと
  "Waiting for background tasks to complete" で停止し、15秒待っても落ちず SIGKILL が必要だった。
  SIGKILL なので pidfile が残り、「立っている」と嘘をつく。**1回のみ観測。**

---

本文書の作成には Claude Opus 5 を用いた。すべての数値はこの箱での実測で、ローカルのログに辿れる。
