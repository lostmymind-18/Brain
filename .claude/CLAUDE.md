# CLAUDE.md — BookMind Tutor

## Dự án là gì

BookMind Tutor: AI gia sư tương tác dựa trên sách, kết hợp RAG + Agent Harness + Knowledge Graph + RL fine-tuning (DPO proof-of-concept). Đây là **dự án học tập cá nhân kéo dài 13 tuần**, mục tiêu chính là:

1. Nắm vững kỹ năng Agent Engineering (harness, reasoning strategies, memory, KG)
2. Có một proof-of-concept RL/DPO tử tế để đưa vào portfolio
3. Sản phẩm cuối cùng dùng để trình bày trong phỏng vấn

Roadmap đầy đủ nằm ở `docs/BookMind_Tutor_Roadmap.md` — luôn tham khảo file đó để biết tuần hiện tại đang làm gì, learning objectives, và evaluation criteria. Nếu prompt của tuần và roadmap có mâu thuẫn, hỏi lại thay vì tự suy đoán.

**Giá trị cốt lõi (áp dụng xuyên suốt mọi tuần, không chỉ riêng phần tutor):**
- Tra cứu/tổng hợp nhanh từ sách (RAG + KG) là giá trị **chắc chắn phải đạt được** — đây là phần rủi ro thấp, lợi ích rõ ràng, không được đánh đổi để làm phần khác.
- Việc đào sâu kiến thức phụ thuộc phần lớn vào sự tò mò/kiên trì của user, không phải vào việc agent "dạy giỏi" đến đâu. Agent chỉ đóng vai trò phản hồi tốt và gợi mở đúng lúc — đây là phần thưởng cộng thêm, không phải điều kiện thành công của dự án.
- Khi thiết kế bất kỳ phần nào liên quan đến trải nghiệm học (UI, evaluation, tutor logic), ưu tiên tối ưu cho "trả lời tốt câu hỏi được hỏi" hơn là cố xây dựng cơ chế "tự động thúc đẩy user học sâu hơn".

## Nguyên tắc làm việc quan trọng

- **Đây là dự án học tập, không phải chạy nước rút ra sản phẩm.** Ưu tiên code dễ hiểu, có comment giải thích *tại sao* chứ không chỉ *làm gì* — vì mục tiêu là học, không chỉ là có code chạy được.
- **Trung thực về scope và giới hạn.** Đặc biệt ở phần RL (Tuần 10-11): không phóng đại là "RLHF hoàn chỉnh" hay "agent tự học online" khi thực chất là offline DPO proof-of-concept trên dataset semi-synthetic. Ghi rõ giới hạn trong code comment và docstring khi liên quan.
- **Nếu một task có rủi ro thời gian cao (theo bảng rủi ro trong roadmap), ưu tiên giải pháp đơn giản chạy được, để lại note TODO cho phần nâng cao** thay vì cố làm hoàn hảo ngay từ đầu. Ví dụ: Tuần 1 nếu table/image extraction tốn thời gian → bỏ qua, chỉ lo text pipeline.
- **Không tự ý mở rộng phạm vi ngoài prompt của tuần đó.** Nếu thấy có việc nên làm nhưng nằm ngoài phạm vi, ghi vào TODO hoặc hỏi trước, đừng tự động implement.
- **Agent là reactive (trả lời tốt), không phải proactive (tự dẫn dắt/lên giáo án).** TutorAgent/QAAgent không tự thiết kế lộ trình học hay chủ động cầm trịch cuộc trò chuyện. Vai trò chính: trả lời chính xác, có chiều sâu khi user đào sâu câu hỏi. Socratic method, Feynman technique là kỹ thuật mix vào câu trả lời khi phù hợp — không phải luồng chính bắt buộc hay cơ chế "thúc đẩy" user. Nguyên tắc này áp dụng khi thiết kế bất kỳ phần nào chạm đến trải nghiệm học (không chỉ Tuần 5-6) — ví dụ evaluation (Tuần 7) nên đo chất lượng phản hồi, không đo "khả năng dẫn dắt chủ động".

## Tech stack đã chốt

- **Ngôn ngữ:** Python 3.11+
- **PDF parsing:** PyMuPDF (`pymupdf`) — bản AGPL open-source, đủ dùng cho dự án học tập phi thương mại. Không cần PyMuPDF Pro.
- **Sentence splitting:** nltk hoặc spaCy (ưu tiên cái nào setup nhẹ hơn cho từng task cụ thể)
- **Testing:** pytest, ưu tiên unit test với dữ liệu giả/mock hơn là phụ thuộc fixture file lớn
- **Data models:** dùng `@dataclass` làm mặc định; chỉ dùng pydantic nếu cần validation phức tạp và phải giải thích lý do trong PR/commit message
- **Frontend (Tuần 5-6):** Streamlit - Python-native, có sẵn file upload + chat UI, không cần frontend riêng. Đủ để demo và interview.
- **Structured logging (Tuần 8-9):** `structlog` — keyword-argument style, JSON output trong production (`LOG_FORMAT=json`), colored pretty output trong dev.
- **Distributed tracing (Tuần 8-9):** OpenTelemetry SDK — `opentelemetry-api`, `opentelemetry-sdk`, `opentelemetry-exporter-otlp-proto-grpc`. Export console (default) hoặc OTLP/gRPC đến Jaeger/Grafana Tempo (`OTEL_EXPORT=otlp`). Xem `docs/local_observability.md` để setup local.
- Các quyết định tech stack khác (framework agent, vector store, LLM provider, graph DB...) — xem thêm `docs/decisions.md` cho lý do đằng sau mỗi lựa chọn.

## Cấu trúc thư mục

```
bookmind-tutor/
  .claude/
    CLAUDE.md
  bookmind_tutor/          # source code chính
    ingestion/             # Tuần 1: PDF → hierarchical chunks
    retrieval/             # Tuần 2: ChromaDB vector store
    agents/                # Tuần 2-3: harness, strategies, tools
    knowledge_graph/       # Tuần 4: entity extraction, Neo4j, GraphRAG
    llm/                   # Tuần 5-6 (refactor): provider abstraction
    tutor/                 # Tuần 5-6: QAAgent, Streamlit UI
    evaluation/            # Tuần 7: InteractionLogger, LLM-as-judge, EvalRunner
    safety/                # Tuần 8-9: guardrails (LengthGuard, PromptInjection, Topicality)
    observability/         # Tuần 8-9: OTel setup, tracing helpers
    rl/                    # Tuần 10-11: DPO fine-tuning (chưa implement)
  tests/
  docs/
    BookMind_Tutor_Roadmap.md
    local_observability.md  # Jaeger Docker setup, OTLP config
  plans/                   # kiến trúc và đề xuất kỹ thuật chưa implement
  PROGRESS.md              # những gì đã implement, reference tới plans/
  backlog.md               # các issue/cải tiến đã phát hiện, kèm trạng thái
  requirements.txt
  lessons/                 # bài học đúc kết trong quá trình implement
```

## Quy trình làm việc

- Bạn và user sẽ tương tác để tìm ra giải pháp. Hoặc bạn sẽ nhận được lệnh implement trực tiếp từ người dùng.
- Trước khi implement, nội dung implement phải được documents trước ở đâu đó, hoặc là trong /plan, hoặc là từ backlog.md. Nếu không có thì phải tạo trước khi implement.
- Sau khi implement, chạy những test cần thiết. Sau khi implement được chấp nhận, cập nhật vào file PROGRESS.md về implement vừa hoàn thành. Với reference tới /plan hoặc backlog.md tương ứng.
- Quy trình làm việc này đảm bảo tiến độ dự án được tracking.
- Sau khi implement hoặc sửa bất kỳ UI component/frontend feature nào, LUÔN dùng Playwright MCP để navigate tới page liên quan, verify render đúng, check console errors, và chụp screenshot trước khi báo "done".

## Coding conventions

- Mỗi module/class có trách nhiệm rõ ràng, tránh "god class". Nếu một class đang làm quá nhiều việc, đề xuất tách trước khi viết tiếp.
- Metadata quan trọng hơn tốc độ ở giai đoạn học — ví dụ: `Chunk`, `Entity` nên giữ đủ thông tin truy vết ngược (page range, char offset, source) vì các tuần sau (KG, evaluation) sẽ cần lại.
- Viết docstring cho public class/method, đặc biệt các heuristic (heading detection, entity extraction) cần giải thích rõ logic và giới hạn đã biết.
- Unit test đi kèm ngay khi viết class mới, không dồn về cuối giai đoạn (roadmap có ghi chú riêng về việc này ở Tuần 8-9).

## Khi không chắc chắn

Nếu prompt không rõ ràng về một quyết định kỹ thuật (vd: chọn thư viện, cấu trúc dữ liệu), hỏi lại thay vì tự chọn — dự án này tôi cần hiểu rõ *lý do* đằng sau mỗi lựa chọn, không chỉ cần code chạy được.

## Ghi lại những bài học thú vị
Khi một thử thách được vượt qua, một giải pháp tốt được thiết kế, điều này có nghĩa là có những kiến thức mới và thú vị đáng được ghi lại. Hãy gợi ý người dùng về việc ghi lại và đúc kết những bài học đó vào trong folder /lessons 
