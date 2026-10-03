# Phân tích kết quả benchmark

Benchmark được chạy offline với estimator token tất định, ngưỡng compact mặc
định `1000` token và giữ lại `4` message gần nhất. Hai agent nhận cùng input;
câu hỏi recall được gửi trong thread mới, nên điểm recall thực sự đo khả năng
nhớ qua thread chứ không phải khả năng đọc lại lịch sử vừa chat.

## Kết quả đo được

### Standard Benchmark

| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 1596 | 14981 | 0.00% | 0.00% | 0 | 0 |
| Advanced | 1710 | 22540 | 100.00% | 100.00% | 343 | 0 |

### Long-Context Stress Benchmark

| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 274 | 22263 | 0.00% | 0.00% | 0 | 0 |
| Advanced | 394 | 11541 | 100.00% | 100.00% | 216 | 3 |

Hai lần chạy liên tiếp trên cùng input cho cùng output. `data/` không bị thay
đổi. `python -m pytest src/test_benchmark.py -q -p no:cacheprovider` cho kết
quả `8 passed`; toàn bộ `python -m pytest src -q -p no:cacheprovider` cho
`40 passed`. Chạy lại bảng bằng `python src/benchmark.py`.

## Bốn kết luận chính

### 1. Advanced recall tốt hơn nhờ persistent profile

Ở cả hai bảng, `Cross-session recall` của Baseline là `0.00%`, còn Advanced là
`100.00%`. `Memory growth (bytes)` của Baseline cũng bằng `0`, trong khi
Advanced tăng `343` bytes ở Standard và `216` bytes ở Stress. Đây là dấu hiệu
đúng với thiết kế: Baseline chỉ giữ `sessions[thread_id]`, không đọc hoặc ghi
`User.md`; thread recall mới vì vậy không có facts cũ.

Ở Advanced, `_reply_offline()` gọi `extract_profile_updates()` trên từng lượt,
ghi các fact bằng `UserProfileStore.upsert_fact()`, rồi `_offline_response()`
đọc lại `User.md` khi trả lời câu hỏi trong thread mới. Vì vậy lợi thế recall
đến từ đường đi cụ thể `extract -> User.md -> read trong thread mới`, không chỉ
đến từ việc Advanced có thêm lịch sử ngắn hạn.

### 2. Advanced đắt hơn khi hội thoại còn ngắn

Trong Standard, Advanced sinh `1710` agent tokens so với `1596` của Baseline và
xử lý `22540` prompt tokens so với `14981`, tức nhiều hơn `7559` prompt tokens.
Ở giai đoạn này chưa có compaction nào (`Compactions = 0`). Mỗi lượt Advanced
phải đưa cả nội dung `User.md` vào phép tính prompt qua
`_estimate_prompt_context_tokens()`, bên cạnh summary và messages gần đây; các
lượt recall cũng tạo câu trả lời đầy đủ hơn. Chi phí profile và recall vì thế
chưa được phần compact bù lại.

### 3. Compact tạo lợi thế rõ ở hội thoại dài

Trong Stress, Baseline xử lý `22263` prompt tokens, còn Advanced chỉ xử lý
`11541`, giảm `10722` tokens, tương đương khoảng `48.2%`. Advanced có
`Compactions = 3`, trong khi Baseline không compact. `CompactMemoryManager.append()`
gộp các message cũ vào summary khi tổng context vượt ngưỡng, rồi chỉ giữ `4`
message gần nhất; `_estimate_prompt_context_tokens()` tính summary và phần
recent đó thay cho toàn bộ lịch sử.

Điểm cần tách bạch là compact tối ưu `Prompt tokens processed`, không phải
`Agent tokens only`: Advanced vẫn sinh `394` agent tokens, cao hơn Baseline
`274`. Compact làm giảm lượng context phải đọc lại qua nhiều lượt; nó không
trực tiếp làm câu trả lời ngắn hơn.

Phép kiểm chứng bổ sung củng cố kết luận này. Khi đặt
`COMPACT_THRESHOLD_TOKENS=10000000`, Advanced Stress tăng lên `23612`
prompt tokens và `Compactions` giảm về `0`. Con số này gần với Baseline
`22263` hơn nhiều so với `11541`, nên lợi thế prompt của Advanced đến từ
compact chứ không phải chỉ từ `User.md`.

### 4. Memory có lợi ích nhưng có chi phí và rủi ro

Advanced tăng file profile `343` bytes ở Standard và `216` bytes ở Stress, đồng
thời Stress thực hiện `3` lần compact. File không tăng ở Baseline vì agent này
không ghi persistent memory. Việc lưu profile giúp recall, nhưng file sẽ tiếp
tục phình khi thêm fact hoặc preference; phần profile này cũng được đưa vào
prompt mỗi lượt.

Có hai rủi ro cụ thể. Thứ nhất, `extract_profile_updates()` là bộ regex tiếng
Việt có chủ đích bảo thủ, nên cách diễn đạt ngoài mẫu có thể bị bỏ sót hoặc bị
hiểu sai. Thứ hai, compact là summary mất thông tin: summary chỉ giữ một số
đoạn ưu tiên, nên chi tiết không được chọn có thể biến mất sau nhiều lần nén.
Một correction mới cũng cần thay đúng fact cũ; nếu luật trích xuất không nhận
ra correction, profile sai có thể được giữ lại.

## Bonus được chọn: Conflict handling khi có correction mới

**Vấn đề có bằng chứng.** Stress dataset cố tình chứa một correction và hai
lượt nhiễu: "Lúc đầu mình nói hiện ở Huế, nhưng thực ra ... đang làm việc ở Đà
Nẵng vài tháng", câu đùa "chuyển sang product manager", và "Hà Nội chỉ là nơi
mình vừa bay ra họp hai ngày". Câu recall thứ hai hỏi thẳng vào xung đột này:
"Nếu ai đó nhắc Huế, Hà Nội hay product manager, đâu mới là nghề nghiệp và nơi ở
hiện tại của mình?". Một extractor ngây thơ kiểu "fact mới nhất thắng" sẽ ghi
`location: Hà Nội` hoặc `profession: product manager` và trả lời sai.

**Cơ chế đã cài.** Ba lớp trong `src/`:

1. `extract_profile_updates()` cắt câu tại `nhưng thực ra|nhưng hiện tại` và chỉ
   giữ vế sau, nên vế lịch sử "hiện ở Huế" không bị ghi.
2. Câu có `?`, `nếu`, `giả sử`, `đùa`, `tạm thời`, `lúc đầu`, `trước đây`,
   `hồi trước` hoặc yêu cầu "nhắc lại/nhớ lại" bị bỏ qua, nên câu hỏi và câu nhiễu
   không trở thành fact. Nơi làm việc chỉ được coi là nơi ở khi có tín hiệu
   chuyển/ở lại rõ ràng (`vài tháng`, `chuyển đến`).
3. `UserProfileStore.upsert_fact()` thay đúng dòng `- key:` cũ và xóa bản trùng,
   nên correction ghi đè thay vì nối thêm. Với `response_style`,
   `_reply_offline()` gộp các preference độc lập nhưng số bullet mới thay số cũ.

**Cải thiện recall/token thế nào.** `User.md` sau stress run chỉ có một giá trị
cho mỗi key:

```markdown
# User profile
- name: DũngCT Stress
- location: Đà Nẵng
- profession: MLOps engineer
- response_style: ngắn gọn, trade-off, ví dụ thực chiến, có cấu trúc, 3 bullet
- interests: Python, AI, MLOps
```

Không có Huế, Hà Nội hay product manager, nên câu recall về xung đột đạt đủ
`MLOps engineer` và `Đà Nẵng`, góp vào `Cross-session recall = 100.00%` ở bảng
Stress. Vì fact cũ bị thay thế chứ không bị nối thêm, `Memory growth` chỉ là
`216` bytes cho 16 lượt dài, và phần profile mang vào prompt mỗi lượt không phình
theo số lần sửa. Hành vi này được khóa bằng test:
`test_questions_and_noise_are_not_facts` (8 câu nhiễu/câu hỏi trả `{}`) và
`test_explicit_facts_and_corrections` trong `src/test_memory_store.py`.

**Rủi ro thêm vào.** Đây là luật regex, nên đánh đổi là false negative và giòn:

- Một fact thật nằm trong câu có chữ `nếu` hoặc có dấu `?` (ví dụ "Mình ở Đà Nẵng,
  bạn biết chỗ nào ngon không?") sẽ bị bỏ qua toàn câu.
- Correction diễn đạt ngoài mẫu (`thật ra`, `giờ thì`, "không phải Huế mà là ...")
  không được nhận, nên fact cũ sai vẫn bị giữ.
- Ghi đè là mất lịch sử: không còn biết trước đây user ở Huế, và một lượt nhiễu
  lọt qua bộ lọc sẽ xóa luôn fact đúng mà không có bước xác nhận.

Rủi ro cuối cùng là lý do của bonus thứ hai bên dưới.

## Bonus thứ hai: Confidence threshold trước khi ghi `User.md`

**Vấn đề.** Conflict handling chỉ lọc theo mẫu câu: một câu không có `?`, `nếu`
hay `đùa` nhưng vẫn mơ hồ ("Hình như mình làm product manager", "Có lẽ mình ở
Hà Nội") vẫn qua regex và ghi thẳng vào `User.md`. Tệ hơn, một câu bình thường
như "Mình ở Đà Lạt" lọt vào giữa hội thoại sẽ ghi đè `location` đúng mà không
cần bằng chứng gì mạnh hơn lần ghi đầu tiên.

**Cơ chế đã cài.**

- `fact_confidence()` trong `src/memory_store.py` chấm điểm câu chứa fact:
  `0.3` nếu có từ rào đón (`hình như`, `có lẽ`, `chắc là`, `dự định`,
  `đang cân nhắc`, `sắp`); `0.9` nếu có dấu hiệu khẳng định hiện tại hoặc
  correction (`hiện tại`, `bây giờ`, `giờ`, `vẫn`, `đang`, `không còn`,
  `nhưng thực ra`); còn lại `0.7`. `scored_profile_updates()` trả mỗi fact kèm
  điểm của chính câu chứa nó, nên một câu rào đón không kéo điểm của câu khác
  trong cùng lượt.
- `_reply_offline()` chỉ ghi khi điểm `>= profile_confidence_threshold`
  (`0.6`, trường mới trong `LabConfig`). Ghi đè một giá trị khác đã có cần
  thêm `+0.2`, tức phải có dấu hiệu khẳng định/correction. `response_style` được
  gộp chứ không ghi đè, nên phần bổ sung style không bị tính là ghi đè.

Hệ quả: fact rào đón không bao giờ được ghi; fact mới nói bình thường vẫn được
ghi; nhưng muốn đổi fact cũ thì user phải nói rõ ("Hiện tại mình ở Đà Nẵng").
Test `test_confidence_threshold_blocks_hedged_and_weak_overwrites` trong
`src/test_advanced.py` khóa đúng các hành vi quan sát được: "Hình như mình làm
product manager" không tạo `profession`; "Có lẽ mình ở Hà Nội" và "Mình ở Đà Lạt"
không đổi `location: Huế`; "Hiện tại mình ở Đà Nẵng" đổi được và thread mới trả
lời Đà Nẵng, không còn Huế; đặt ngưỡng `0.0` thì fact rào đón lại bị ghi.

**Cải thiện recall/token thế nào.** Trên hai bộ benchmark, bảng kết quả **không
đổi** (recall `100%`, `343`/`216` bytes, prompt `22540`/`11541`). Đây là kết
quả mong muốn nhưng cần đọc đúng: dữ liệu hiện tại không có câu fact rào đón
nào, nên threshold không có gì để chặn; điều bảng chứng minh là nó **không gây
false negative** trên 117 lượt thật của hai bộ dữ liệu. Lợi ích nằm ở trường hợp test mô tả: chặn
fact mơ hồ thì recall không trả lời sai, và `User.md` không phình thêm dòng sai
(cũng là phần được nạp vào prompt mỗi lượt). Muốn đo bằng số, cần thêm một bộ
dữ liệu có câu rào đón; `data/` là input chung nên không được sửa để tạo bộ đó.

**Rủi ro thêm vào, có bằng chứng.** Bản đầu tiên áp `+0.2` cho mọi key và đã
gây false negative thật: khi chạy trên benchmark, nó từ chối 7 lần bổ sung
`response_style`, làm mất "trade-off" ở Standard và "có cấu trúc" ở Stress.
Recall vẫn `100%` chỉ vì câu hỏi recall không hỏi các phần đó, tức bảng
benchmark không tự phát hiện lỗi này. Phải đọc lại `User.md` mới thấy, và sửa
bằng cách loại `response_style` khỏi luật ghi đè. Bài học: threshold làm hệ
thống bảo thủ hơn, và bảo thủ sai chỗ sẽ âm thầm làm mất fact đúng. Các rủi ro
còn lại:

- User đổi nơi ở bằng câu bình thường ("Mình ở Đà Lạt" khi thực sự đã chuyển)
  sẽ bị bỏ qua. Không có bước hỏi lại để xác nhận.
- Điểm số là từ vựng chỉnh tay. "đang" được tính là khẳng định, nên "Mình
  đang ở Hà Nội họp hai ngày" vẫn đủ điểm ghi đè nếu regex trích được.
- Thêm một tham số cần hiệu chỉnh. Ngưỡng quá cao tăng false negative, quá thấp
  quay về hành vi cũ.

## Đối chiếu rubric

- **0-60:** Có Baseline chỉ nhớ theo thread, Advanced có `User.md`, compact,
  dataset benchmark và hai bảng kết quả.
- **60-75:** Hai agent chạy cùng input; benchmark có đủ sáu metric; test hiện
  có các kiểm tra profile, compact trigger, cross-session recall và benchmark
  repeatability.
- **75-90:** Có Standard và Long-Context Stress. Số liệu cho thấy Advanced có
  thể đắt hơn ở Standard (`22540` so với `14981` prompt tokens), nhưng giảm
  mạnh ở Stress (`11541` so với `22263`); lợi thế nằm ở prompt chứ không phải
  agent output.
- **90-100:** Bonus Conflict handling đã được cài và test: giải quyết việc fact
  cũ hoặc nhiễu ghi đè fact hiện tại, giữ recall `100%` ở câu hỏi xung đột của
  Stress với `User.md` chỉ `216` bytes, và nêu rõ rủi ro false negative, regex
  giòn, mất lịch sử khi ghi đè. Bonus Confidence threshold cũng đã được cài và
  test: chặn fact rào đón và ghi đè yếu, không gây false negative trên
  benchmark, và nêu rủi ro dựa trên false negative thật đã gặp với
  `response_style`.

## Kết luận

Kết quả cho thấy một trade-off nhất quán: Baseline rẻ hơn về memory nhưng không
nhớ qua thread; Advanced đổi thêm storage và chi phí prompt ở hội thoại ngắn để
có recall `100%`, sau đó dùng compact để giảm gần một nửa prompt cost ở hội
thoại dài. Hệ thống mạnh hơn vì có persistent memory và bounded context, nhưng
cũng cần guardrail ở bước ghi fact và cần chấp nhận nguy cơ mất chi tiết do
summary nén.
