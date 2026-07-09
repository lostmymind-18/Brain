# Reactive Planning vs Proactive Planning trong Agent Reasoning

## Insight

ReAct cũng có "plan" - nhưng là plan cục bộ, từng bước một.
Plan-and-Solve tách planning thành một bước độc lập trước khi làm bất cứ gì.

Sự khác biệt cốt lõi:

- **ReAct**: liên tục refine plan dựa trên observation vừa nhận được.
  Mỗi Thought chỉ quyết định action *tiếp theo*, không nhìn toàn bộ bức tranh.

- **Plan-and-Solve**: enumerate toàn bộ sub-tasks *trước* khi execute.
  Sau khi có plan, chỉ việc chạy theo - không cần quyết định thêm gì giữa chừng.

## Khi nào dùng cái nào

**ReAct phù hợp hơn khi:**
- Câu hỏi mơ hồ, không biết cần tìm gì cho đến khi thấy kết quả đầu tiên.
- Bước sau phụ thuộc vào nội dung bước trước (plan không thể biết trước hoàn toàn).
- Câu hỏi đơn giản, chỉ cần 1-2 tool call.

**Plan-and-Solve phù hợp hơn khi:**
- Câu hỏi có nhiều sub-requirements rõ ràng ngay từ đầu.
- Sợ bỏ sót yêu cầu - vì ReAct dễ "quên" một phần câu hỏi khi vòng lặp kéo dài.
- Câu hỏi dạng so sánh, phân tích nhiều chiều.

## Ví dụ minh họa

Câu hỏi: *"Compare leader-based vs leaderless replication - trade-offs, use cases, và ví dụ hệ thống thực tế"*

ReAct có thể search leader → search leaderless → trả lời, nhưng đến lúc tổng hợp
có thể quên mất yêu cầu "ví dụ hệ thống thực tế" vì nó nằm ở cuối câu hỏi.

Plan-and-Solve bắt đầu bằng:
```
1. Tìm cơ chế leader-based replication
2. Tìm cơ chế leaderless replication
3. Tìm trade-offs của từng loại
4. Tìm ví dụ: Kafka, Cassandra, Dynamo...
5. Tổng hợp so sánh
```
Không bỏ sót vì đã enumerate từ đầu.

## Điểm dễ nhầm

Đọc ReAct paper xong dễ nghĩ "ReAct đã có Thought (plan) rồi, Plan-and-Solve thêm gì nữa?"

Câu trả lời: ReAct plan **reactively** (từng bước, sau mỗi observation).
Plan-and-Solve plan **proactively** (toàn bộ, trước mọi action).

## Nguồn

- ReAct: Yao et al. 2022
- Plan-and-Solve: Wang et al. 2023
- Reflexion: Shinn et al. 2023
- Thảo luận phát sinh khi thiết kế Week 3 của BookMind Tutor
