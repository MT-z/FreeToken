# 設計: 会話途中の system メッセージを在置きにする

対象: `python/freetoken/server/anthropic_api.py` の `convert_anthropic_prompt`。
背景と問題の同定は `.claude/REVIEW-prefix-cache.md`（査読2回済み）。**本書は実装前の設計。**

---

## 目的と非目的

**目的。** 会話が伸びてもプロンプトの**先頭が動かない**ようにする。
現状は途中の `role:"system"` を先頭の system ブロックへ巻き上げるため、
ターンごとに約19トークンが先頭側に増え、後続の 80k〜130k が一致しなくなる。

**非目的。**

* GDN スナップショットの配置（査読文書の案C）は扱わない。別件。
* テンプレート制約の回避は扱わない。**先頭の system ブロックは今までどおり先頭に置く。**
* 出力品質の改善は狙わない。**悪化させないことだけを見る。**

---

## 現状の振る舞い（`anthropic_api.py:203-290`）

```
system_texts = [トップレベル system] + [途中の role:"system" を出現順に全部]
messages     = [{"role":"system", "content": "\n\n".join(system_texts)}] + other
```

途中の system は元の位置から消え、先頭に連結される。

---

## 設計の入力（実分布。req-00036、途中 system 24本）

```
  23 本  直後は assistant（thinking+tool_use / text+thinking+tool_use）
   1 本  後続なし
   0 本  直後が user
```

メッセージ内訳（user 24 本）: `tool_result のみ` 22 本、`text` 1 本、`tool_result + text` 1 本。

**ここが設計の分かれ目。** `tool_result` だけの user メッセージは変換で `role:"tool"` に分解され、
**user エントリを1つも生まない**。だから「直後の user ターンにマージ」は、
reminder を **assistant ターンを飛び越えて1ターン後ろへ動かす**ことになる。
査読文書の案A（＝下表の `next`）はこれであり、**位置として忠実ではない。**

### 置き方を3通り測った（58組、同一セッション、GPU 不使用）

| 置き方 | 共通/総 の合計比 | 99%以上の組 | ターン位置 |
|---|---:|---:|---|
| 現状（巻き上げ） | 53.0% | 18/58 | — |
| `next` 直後の user へ前置 | **96.2%** | 33/58 | 1ターン後ろへ動く |
| `prev` 直前の user へ後置 | **96.2%** | 33/58 | 保たれる |
| `own` その場で独立 user ターン | **96.2%** | 33/58 | 保たれる |

一次出力: `freetoken-systest/results/20260907T015806-diag_prefix-cache-placement-3way.json`
道具: `freetoken-systest/tools/prefix-hoist-compare.py --modes next,prev,own`（a0e0d07 以降。それより前の `main` は `next` しか測らなかった）

**「3通り同じ」なのは合計比の話であって、組ごとには一致しない。**

```
3通りの共通長が完全に一致する組 :  1 / 58
own と prev の差                : 最大    4 トークン
own と next の差                : 最大 1,834 トークン
```

つまり `own` と `prev` は実質同じもので、`next` だけが別物。
**合計比で見るかぎり効きは置き方に依存しない**ので、選択基準は性能ではなく忠実性になる。

### テンプレートが許す並び（実測）

| 並び（実データに出る順） | 結果 |
|---|---|
| `user(text), user(reminder), assistant` | OK |
| `tool, user(reminder), assistant` | OK |
| 末尾 `user(reminder)` + 生成プロンプト | OK |
| （対照）途中に `system` を残す | NG `System message must be at the beginning.` |

（手組みで最初に NG が出たのは `tool_calls.function.arguments` を JSON 文字列にしたため。
**これは reminder と無関係**で、reminder 抜きの対照も同じく NG になる。
実機ではこの経路を通らない —— `convert_anthropic_prompt` は最後に `render_messages`
（`anthropic_api.py:323` → `generation.py:275`）を通し、そこで `arguments` が dict に戻る。
**検証1の前提「道具と実機は同じ変換」はこれで成立する。**）

---

## 決定

**D1. 置き方は `own`** —— 途中の `role:"system"` を、**その位置で独立した `user` ターン**にする。
効果が3通り同じ以上、位置を保ち、かつクライアントの構造（reminder は独立した注入）に
最も近いものを採る。査読文書の**案B**にあたる。**案A（`next`）は採らない。**

**この設計でふるまいの差が最も出るのは末尾である。** body 60 本のうち **54 本が system で終わる**。
`own` では、生成プロンプトの直前のターンが `user(reminder)` になる。描画は通る
（`</tool_response><|im_end|>` の後に user の reminder、次に `assistant\n<think>`）。
**検証4の目視は、まずこの形を見る。**

**D2. 「最初の非 system メッセージより前」の system は先頭ブロックへ、それ以降は在置き。**
トップレベル `system` は今までどおり先頭。加えて、`messages` の先頭に並ぶ `role:"system"` も
先頭ブロックに入れる。**これを決めないと system ブロックの無いプロンプトが作れてしまう**
（`tests/server/test_anthropic_api.py:132` の fixture は `messages[0]` が system で
トップレベル `system` が無い。`own` を素直に当てると role 列が `["user","assistant","user"]` になる）。
実データでは `messages[0]` が system の body は 60 本中 **0 本**、
トップレベル `system` が無い body は **1 本**。**正しさとテストのための規則であって、
実トラフィックでは通らない枝。**

**D3. 連続する途中 system は1つの user ターンにまとめる。** 区切りは `\n\n`（現状の連結と同じ）。
ターン数の増加を抑え、区切り文字のゆらぎを減らす。
**ただし実データでは連続する途中 system は 60 本中 0 箇所。害は無いが、今のデータでは通らない枝。**

**D4. 空文字は落とす。** 現状の `if t` フィルタと同じ扱い。

**D5. content がブロック配列の system は text ブロックのみ連結。** 既存の `_content_text` を使う。

**D6. `count_tokens` は自動的に一致する。** 変更点は `convert_anthropic_prompt` の1箇所であり、
`/v1/messages` と `/v1/messages/count_tokens` は同じ関数を通る（docstring がそう宣言している）。

**D7. 切り替えは環境変数 `FREETOKEN_SYSTEM_IN_PLACE`。**
段階1では**既定を現状のまま**にし、フラグで新動作にする。
理由は A/B を1つのビルドで測れるようにするため。実機の時間が取れたら段階4で既定を反転する。

---

## 影響する既存テスト

* `tests/server/test_anthropic_api.py:231` `test_convert_hoists_and_merges_system_messages`
  —— role 列 `["system","user","assistant"]` と system が1本であることを固定している。
  **`own` では `["system","user","user","assistant"]` になるので書き直しが必要**（実測）。
  D7 の既定が現状のままなら段階1では落ちないが、**段階4で必ず落ちる。**
  新動作用のテストを段階1で足し、段階4で旧動作のテストを置き換える。
* `tests/server/test_anthropic_api.py:132` `test_convert_system_role_message_and_unknown_block`
  —— `messages[0]` が system でトップレベル `system` が無い fixture。
  **D2 の規則によりこの system は先頭ブロックへ入るので、role 列は現状のまま**
  `["system","assistant","user"]`。D2 を入れなければ `["user","assistant","user"]` になり落ちる。
  **この fixture が D2 の必要性を示している。**

---

### 参考: fixture ごとの role 列（実測）

| fixture | 現状 | `next` | `prev` | `own` |
|---|---|---|---|---|
| `:231`（トップレベル system 有り） | `system,user,assistant` | `system,user,assistant,user` | `system,user,assistant` | `system,user,user,assistant` |
| `:132`（`messages[0]` が system） | `system,assistant,user` | `assistant,user` | `user,assistant,user` | `user,assistant,user` |

`:132` の `next` / `prev` / `own` はいずれも **D2 を入れる前**の値。入れれば先頭に system が戻る。

---

## 検証（何をもって「効いた」とするか）

1. **オフライン（GPU 不要）**: `prefix-hoist-compare.py --modes next,prev,own` で 53.0% → 96.2%、悪化組なし。
   **これは既に測ってある。** 段階 2 では、実装（フラグ ON）の出力が
   同じ `place_in_place(o, "own")` の描画と **body ごとにトークン列で一致**することを確認する。
2. **単体**: 上記2件を含む `tests/server/test_anthropic_api.py` が通る。
3. **実機（唯一の未測定）**: 同一プロンプト列を冷↔冷で流し、**端から端までの時間**と
   `#cached-token` を新旧で比べる。**トークン一致長ではなく時間で示す。**
4. **出力**: 同じ入力に対する応答を新旧で並べ、**明らかな劣化が無いことを目視で1回**見る。
   **見る対象は末尾の reminder**（body 60 本中 54 本がこの形）。
   role が `system` から `user` に変わって直前のターンになるので、差が出るならここに出る。

---

## 段階3の結果（2026-09-07、この箱で実測）

記録済み本文を記録順に再生、冷↔冷、旧→新→旧→新の交互、`max_tokens=16` 固定。
道具: `freetoken-systest/tools/{replay-bodies.py,measure-system-in-place.sh}`。
一次出力: `freetoken-systest/results/20260907T022122-stage3-*.json`。

```
             合計      中央値/件    最大/件   Prefill行     計算       再利用    再利用率
old-1      792.8s      7.73s     34.60s      432   3,236,348  2,175,744   40.2%
old-2      792.0s      7.66s     34.67s      432   3,236,348  2,175,744   40.2%
new-1      121.1s      1.16s     19.61s       87     283,411  5,132,973   94.8%
new-2      117.5s      1.15s     18.54s       87     283,411  5,132,973   94.8%
```

* **端から端まで 6.6倍**（792 → 119 秒）
* **実際に計算した prefill は 11.4分の1**（3,236,348 → 283,411 トークン）
* **再利用率 40.2% → 94.8%**
* 1件ごとの短縮率は中央値 **5.61倍**、最大 26.41倍
* **遅くなった本文は 60 本中 3 本、いずれも +0.14 秒以内**（1.15→1.29 / 1.17→1.30 / 1.16→1.21）。
  ノイズの幅

**再現性。** 旧は 792.8 対 792.0 秒（差 0.1%）、新は 121.1 対 117.5 秒（差 3%）。
Prefill の行数・計算量・再利用量は**繰り返しの間で完全に同一**（決定的）。

**本数について。** オフライン比較は 59 本（`req-00018` を `/v1/messages` として読めず除外）だったが、
再生では **60 本**。再生側は `max_tokens` を注入するため、`count_tokens` 用の本文も
有効な `/v1/messages` 要求になる。**両者で母数が1本違う。**

**注意。** これは記録済みの本文をそのまま投げ直した数字であり、
**対話の体感時間そのものではない**。`max_tokens=16` で decode を切っているので、
実際の利用では decode 時間がこれに乗る。**測ったのは prefill の再利用である。**

## 決めていないこと

* **出力品質。** role が `system` から `user` に変わるので reminder の効き方は変わりうる。
  **測るより決める話**であり、本設計では「悪化させない」以上の主張をしない。
* **他機種。** 途中 system を許すテンプレートでは巻き上げ自体が不要かもしれない。
  テンプレート能力で分岐すべきかは決めていない。D7 のフラグはその判断を先送りできる形にしてある。
* ~~**96.2% で頭打ちになる理由。**~~ —— **査読で解けた。キャッシュの欠陥ではない。**
  B は A に新しいターンを足したものなので、一致率は「足したターンの大きさ」で決まる。
  尾（総 − 共通）の中央値は、99% に届く 33 組が **191 トークン**、届かない 25 組が **2,321 トークン**。
  尾が大きい4組の正体は、大きな `tool_result` を足した組（18,403 / 9,657）、
  除外した `req-00018` をまたぐ組（29,323）、別会話の境界（74,865）。**未解決項目から外す。**
  （数値はすべて `next` 列。D1 の `own` では 190 / 2,246、4 組は 18,398 / 9,537 / 29,318 / 74,869 で同傾向。）

---

## 段階

1. `own` を `FREETOKEN_SYSTEM_IN_PLACE` の裏に実装。新動作のテストを足す。既定は現状のまま
2. オフライン比較（`tools/system-in-place-stage2.py`）で、実装の出力が測定済みの `own` 変換と body ごとに一致することを確認
3. 実機で新旧の時間を測る（冷↔冷）。出力を1回見る
4. 数字が出たら既定を反転し、旧動作のテストを置き換える

### 進捗（2026-09-07）

* **段階 1 済み。** 枝 `feat/system-in-place`（`diag/prefix-cache` から）。`convert_anthropic_prompt` に
  `FREETOKEN_SYSTEM_IN_PLACE` を読む分岐を足し、D1〜D5 をそのまま実装。既定は現状のまま。
  新動作のテスト 6 本を `tests/server/test_anthropic_api.py` に追加（既定 OFF、`own` の role 列、
  D2 の先頭 system、D3/D4 の連結と空文字、末尾 reminder、D6 の count_tokens 一致）。
* **段階 2 済み。** `tools/system-in-place-stage2.py`（systest）で、フラグ ON の実装の描画と
  `place_in_place(o, "own")` → フラグ OFF の描画を body ごとに比べた。
  **59 本中 59 本でトークン列が同一。** 連続 58 組の合計比 96.2%、99% 以上 33 組、
  100 トークン超の悪化 0 組で、3 通り計測の `own` 列と一致する。
  一次出力: `freetoken-systest/results/20260907T021343-diag_system-in-place-stage2.json`
* **段階 3 は未着手（GPU）。** 段階 4 はその数字を待つ。

---

本設計の作成には Claude Opus 5、査読と修正・実装には Claude Fable 5.1 を用いた。数値はこの箱での実測。
