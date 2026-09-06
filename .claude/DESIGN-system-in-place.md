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

メッセージ内訳: `user / tool_result のみ` が 22 本、`user / text` は 1 本のみ。

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

**キャッシュの効きは置き方に依存しない。** だから選択基準は性能ではなく忠実性になる。

### テンプレートが許す並び（実測）

| 並び | 結果 |
|---|---|
| `user(reminder), user` | OK |
| `user(reminder), assistant` | OK（実分布の多数） |
| `user(reminder), tool` | OK |
| `tool, user(reminder), assistant` | OK |

（初回 NG が出たのは `tool_calls.function.arguments` を JSON 文字列で組んだため。
このテンプレートは dict を要求する。reminder の位置とは無関係で、対照も同じく NG になった。
**FreeToken の変換は `json.dumps` で文字列を入れている**が、実機は動いているので
どこかで整合しているはず。**確認していない。本設計の範囲外。**）

---

## 決定

**D1. 置き方は `own`** —— 途中の `role:"system"` を、**その位置で独立した `user` ターン**にする。
効果が3通り同じ以上、位置を保ち、かつクライアントの構造（reminder は独立した注入）に
最も近いものを採る。査読文書の**案B**にあたる。**案A（`next`）は採らない。**

**D2. 先頭の system ブロックは変えない。** トップレベル `system` は今までどおり先頭に置く。
テンプレートの `System message must be at the beginning.` を満たす。

**D3. 連続する途中 system は1つの user ターンにまとめる。** 区切りは `\n\n`（現状の連結と同じ）。
ターン数の増加を抑え、区切り文字のゆらぎを減らす。

**D4. 空文字は落とす。** 現状の `if t` フィルタと同じ扱い。

**D5. content がブロック配列の system は text ブロックのみ連結。** 既存の `_content_text` を使う。

**D6. `count_tokens` は自動的に一致する。** 変更点は `convert_anthropic_prompt` の1箇所であり、
`/v1/messages` と `/v1/messages/count_tokens` は同じ関数を通る（docstring がそう宣言している）。

**D7. 切り替えは環境変数 `FREETOKEN_SYSTEM_IN_PLACE`。**
段階1では**既定を現状のまま**にし、フラグで新動作にする。
理由は A/B を1つのビルドで測れるようにするため。実機の時間が取れたら段階2で既定を反転する。

---

## 影響する既存テスト

* `tests/server/test_anthropic_api.py:231` `test_convert_hoists_and_merges_system_messages`
  —— role 列 `["system","user","assistant"]` と system が1本であることを固定している。
  **新動作では `["system","user","assistant","user"]` になるので書き直しが必要。**
  D7 の既定が現状のままなら段階1では落ちないが、**段階2で必ず落ちる。**
  新動作用のテストを段階1で足し、段階2で旧動作のテストを置き換える。
* `tests/server/test_anthropic_api.py:132` `test_convert_system_role_message_and_unknown_block`
  —— 途中 system を含む。アサーションの確認が要る。

---

## 検証（何をもって「効いた」とするか）

1. **オフライン（GPU 不要）**: `prefix-hoist-compare.py` で 53.0% → 96.2%、悪化組なし。
   **これは既に測ってある。実装がこの変換と一致することの確認に使う。**
2. **単体**: 上記2件を含む `tests/server/test_anthropic_api.py` が通る。
3. **実機（唯一の未測定）**: 同一プロンプト列を冷↔冷で流し、**端から端までの時間**と
   `#cached-token` を新旧で比べる。**トークン一致長ではなく時間で示す。**
4. **出力**: 同じ入力に対する応答を新旧で並べ、**明らかな劣化が無いことを目視で1回**見る。

---

## 決めていないこと

* **出力品質。** role が `system` から `user` に変わるので reminder の効き方は変わりうる。
  **測るより決める話**であり、本設計では「悪化させない」以上の主張をしない。
* **他機種。** 途中 system を許すテンプレートでは巻き上げ自体が不要かもしれない。
  テンプレート能力で分岐すべきかは決めていない。D7 のフラグはその判断を先送りできる形にしてある。
* **96.2% で頭打ちになる理由。** 33/58 組が 99% 以上に届く一方、残りは届かない。
  原因を追っていない。**相関のまま置く。**

---

## 段階

1. `own` を `FREETOKEN_SYSTEM_IN_PLACE` の裏に実装。新動作のテストを足す。既定は現状のまま
2. オフライン比較で、実装の出力が測定済みの `own` 変換と一致することを確認
3. 実機で新旧の時間を測る（冷↔冷）。出力を1回見る
4. 数字が出たら既定を反転し、旧動作のテストを置き換える

---

本設計の作成には Claude Opus 5 を用いた。数値はこの箱での実測。
