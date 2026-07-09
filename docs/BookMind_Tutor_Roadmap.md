# BookMind Tutor — Roadmap Học & Xây Dựng

*Building Advanced Agentic Learning Systems (+ Knowledge Graph + RL Fine-tuning)*

**Dự án:** BookMind Tutor – AI Gia sư tương tác + Knowledge Graph + Advanced RL Fine-tuning

**Thời lượng:** 13 tuần (mở rộng từ 10 tuần để phần RL/DPO có đủ không gian làm nghiêm túc)

**Mục tiêu:** Xây dựng một hệ thống agentic mạnh, có portfolio chất lượng cao, và nắm vững kỹ năng Agent Engineering + RL cơ bản cho LLM.

**Giá trị cốt lõi của dự án (tại sao làm):** Đây là dự án bản lề giúp tăng tốc việc học các chủ đề khó trong tương lai — không chỉ qua kỹ năng kỹ thuật tích lũy được (RAG, agent, KG, evaluation, DPO — tái sử dụng cho mọi dự án AI sau này), mà còn qua chính công cụ tạo ra: tra cứu/tổng hợp nhanh từ sách là giá trị *chắc chắn đạt được*; việc đào sâu kiến thức phụ thuộc phần lớn vào sự tò mò của người dùng, agent chỉ đóng vai trò phản hồi tốt và gợi mở đúng lúc — đây là *phần thưởng cộng thêm*, không phải điều kiện thành công của dự án.

---

# Tuần 0 (Chuẩn bị — khuyến nghị thêm)

☐ Chốt tech stack: framework (LangChain / LlamaIndex / tự viết), LLM provider, vector store

☐ Chuẩn bị sách/tài liệu mẫu (ưu tiên public domain để tránh vấn đề bản quyền)

☐ Setup repo, môi trường, CI cơ bản

☐ Định nghĩa "definition of done" cho từng tuần

# Tuần 1: Document Intelligence & Ingestion Pipeline

**Learning Objectives**
- Xử lý PDF chuyên sâu và chunking thông minh
- Hiểu rõ RAG pipeline
- Xây dựng ingestion system robust

**Key Concepts**
- PDF parsing (text, tables, images)
- Hierarchical chunking (chapter → section → paragraph)
- Embedding models & vector stores

**Coding Tasks**
☐ Xây PDF uploader + text extractor
☐ Implement hierarchical chunker
☐ Basic RAG chat với sách

**Reading Materials**
- Advanced RAG techniques (LlamaIndex / LangChain docs)
- "Chunking Strategies for Long Documents"

**Evaluation Criteria**
- Xử lý được sách > 300 trang
- Chunk quality tốt (không cắt giữa câu)
- Chat cơ bản mượt mà

**⚠** Nếu table/image extraction tốn quá nhiều thời gian, hạ xuống stretch goal, ưu tiên text pipeline chạy tốt trước.

# Tuần 2: Agent Harness Foundations

**Learning Objectives**
- Hiểu và tự implement Agent Harness
- Quản lý state và memory

**Key Concepts**
- Agent loop architecture
- Tool calling system
- Memory types (conversation, episodic, semantic)

**Coding Tasks**
☐ Xây AgentHarness class
☐ Implement ReAct loop cơ bản
☐ Tool executor + error handling

**Reading Materials**
- CMU AI Agents course (schedule)
- ReAct paper + LangGraph core concepts

**Evaluation Criteria**
- Harness chạy ổn định, có logging
- Xử lý lỗi tốt (retry, fallback)

# Tuần 3: Advanced Reasoning Strategies

**Learning Objectives**
- Implement nhiều reasoning patterns

**Key Concepts**
- ReAct vs Plan-and-Execute vs Reflexion
- Structured output & parsing
- Dynamic routing

**Coding Tasks**
☐ Implement 3 strategies khác nhau trên cùng harness
☐ Thêm Plan-and-Execute + Reflexion
☐ Dynamic strategy routing

**Reading Materials**
- Reflexion paper, Plan-and-Solve, ReWOO

**Evaluation Criteria**
- So sánh hiệu suất giữa các strategies trên cùng task

# Tuần 4: Knowledge Graph & Long-term Memory

**Learning Objectives**
- Xây dựng và vận hành Knowledge Graph

**Key Concepts**
- Entity & Relation extraction
- Graph RAG
- Dynamic knowledge updating

**Coding Tasks**
☐ Setup Neo4j
☐ Entity extraction từ sách + conversation
☐ Graph RAG retriever

**Reading Materials**
- GraphRAG papers (Microsoft)
- Neo4j + LangChain integration

**Evaluation Criteria**
- Knowledge graph được xây dựng tự động và có ý nghĩa

**⚠** Rủi ro thời gian cao nhất trong phần agent engineering. Nếu sau 5-6 ngày entity extraction chưa ổn, hạ tiêu chuẩn xuống các loại entity cố định (khái niệm, định nghĩa, nhân vật...) thay vì để mở tự do.

**✅ Checkpoint tích hợp:** Cuối tuần này, chạy thử pipeline end-to-end (ingestion → harness → reasoning → KG) trước khi thêm lớp pedagogical.

# Tuần 5–6: Responsive Q&A Agent (với kỹ thuật sư phạm hỗ trợ)

> **Cập nhật khung tư duy (quan trọng):** Agent ở đây về bản chất là **reactive Q&A agent**, không phải một gia sư chủ động (proactive tutor) tự thiết kế giáo án hay lộ trình học cho user đi theo. Vai trò chính: **trả lời câu hỏi của user thật tốt**. Socratic method và Feynman technique là **kỹ thuật tăng cường câu trả lời**, được mix vào khi phù hợp — không phải kiến trúc cốt lõi hay chức năng chính của agent. Việc đào sâu kiến thức đến đâu phụ thuộc phần lớn vào sự tò mò/kiên trì của user; agent chỉ có nhiệm vụ phản hồi tốt và gợi mở đúng lúc khi có thể, không tự khởi xướng một lộ trình giảng dạy dài hạn.

**Learning Objectives**
- Xây dựng agent phản hồi chính xác, có chiều sâu khi user đào sâu câu hỏi
- Tích hợp kỹ thuật Socratic questioning và Feynman check như công cụ hỗ trợ câu trả lời (không phải luồng chính bắt buộc)
- Student modeling ở mức nhẹ: điều chỉnh độ sâu/độ khó câu trả lời theo cách user đặt câu hỏi

**Key Concepts**
- Socratic method như một kỹ thuật phản hồi (hỏi ngược khi phù hợp, không phải mặc định)
- Feynman technique như công cụ kiểm tra hiểu biết theo yêu cầu hoặc gợi ý đúng lúc
- Adaptive response depth (không phải adaptive curriculum)
- Student modeling nhẹ — nắm được level hiểu biết hiện tại của user qua cách họ hỏi, không xây dựng hồ sơ học tập dài hạn phức tạp

**Coding Tasks**
☐ Xây QAAgent class (đổi tên từ TutorAgent để phản ánh đúng bản chất reactive)
☐ Implement cơ chế mix Socratic questioning / Feynman check vào câu trả lời khi ngữ cảnh phù hợp (không phải fixed workflow)
☐ Adaptive response depth dựa trên câu hỏi/phản hồi của user
☐ Student modeling nhẹ (nhận diện level hiện tại qua ngôn ngữ, độ chi tiết câu hỏi)

**User Knowledge Graph (Personal Brain)**

KG không phải index của sách — KG là "brain" của user, mô hình tri thức cá nhân tiến hoá theo thời gian tương tác.
Thiết kế đầy đủ: `plans/user_knowledge_graph.md`.

☐ `KGUpdateSuggester`: sau mỗi turn, extract entities từ retrieved chunks, so sánh với user KG, trả về suggestions
☐ `GraphStore` user entity methods: `upsert_user_entity`, `get_user_entity_names`, `get_user_related_entities`
☐ Update `GraphRAGRetriever` dùng user KG để personalize retrieval
☐ Streamlit: suggest → user approve → entities được thêm vào KG

**Frontend (Streamlit)**
☐ Trang Upload: upload PDF → chạy ingestion pipeline → thông báo sẵn sàng
☐ Trang Chat: chat interface dùng `st.chat_message`, hiển thị strategy và số LLM calls mỗi turn
☐ Sidebar: chọn strategy (Auto / ReAct / Plan-and-Execute / Reflexion), hiển thị user KG hiện tại
☐ Hiển thị sources (chapter, page) dưới mỗi câu trả lời
☐ Sau mỗi câu trả lời: hiển thị suggestions "Add to my knowledge graph" với checkbox per entity

Lý do chọn Streamlit: Python-native (nhất quán với stack), có sẵn chat + file upload component, build nhanh, đủ đẹp để demo interview. Không cần React hay FastAPI ở giai đoạn này.

**Reading Materials**
- Papers về AI Tutors, Educational Agents (đọc để lấy ý tưởng kỹ thuật, không cần áp dụng mô hình "proactive curriculum design")

**Evaluation Criteria**
- Agent trả lời chính xác và có chiều sâu khi user tiếp tục đào sâu một chủ đề
- Biết chêm câu hỏi Socratic hoặc gợi ý Feynman check đúng lúc, đúng ngữ cảnh — không lạm dụng, không ép vào mọi câu trả lời
- Không cần đo "khả năng dẫn dắt chủ động" — đây không phải mục tiêu thiết kế
- UI: upload → chat hoạt động end-to-end, sources hiển thị đúng

**💡** Vì tiêu chí đã cụ thể hơn (đo chất lượng phản hồi, không đo khả năng dẫn dắt), rủi ro "khó đo lường khách quan" ở bản roadmap gốc phần lớn đã được giải quyết bằng cách định nghĩa lại đúng vai trò agent.

# Tuần 7: Evaluation Framework & Self-Improvement

**Learning Objectives**
- Đo lường và cải thiện chất lượng agent

**Key Concepts**
- Agent evaluation methodologies (comprehension, retention, engagement)
- Knowledge retention metrics
- Self-reflection loop

**Coding Tasks**
☐ Xây evaluation suite (quiz generation, comprehension score, engagement metrics)
☐ Reflection & improvement loop

**Reading Materials**
- CMU AI Agents assignments (evaluation part)

**Evaluation Criteria**
- Có metric rõ ràng đo chất lượng tutor

**💡** Dữ liệu evaluation ở tuần này (interaction logs, comprehension scores) sẽ là nguồn dữ liệu quan trọng cho phần RL ở Tuần 10-11 — nên log cẩn thận từ bây giờ.

# Tuần 8–9: Production, Safety & User Experience

**Learning Objectives**
- Đưa agent lên mức production

**Key Concepts**
- Safety & guardrails
- Observability (LangSmith hoặc custom)
- Nice UI/UX

**Coding Tasks**
☐ Thêm safety layer
☐ Build UI với Streamlit/Gradio
☐ Experiment tracking cơ bản

**Reading Materials**
- Agent safety best practices

**Evaluation Criteria**
- Hệ thống chạy ổn định, an toàn, UI thân thiện

**💡** Nên viết unit test tối thiểu (tool executor, chunker...) từ tuần 2 thay vì dồn hết vào giai đoạn này.

**✅** Đây là bản "hoàn chỉnh mức production" của hệ thống — dùng UI này để thu thập interaction data thực tế cho Tuần 10-11.

# Tuần 10–11: Advanced RL & Fine-tuning (Module mở rộng — Proof-of-Concept)

**Learning Objectives**
- Sử dụng interaction data để cải thiện agent một cách có hệ thống
- Hiểu và áp dụng RL cơ bản cho Agentic Systems

**Khung thực tế (quan trọng):** Với quy mô 1 người trong ~2 tuần, mục tiêu khả thi là một **offline DPO proof-of-concept** trên model nhỏ — không phải online RL hay reward model quy mô lớn. Đây vẫn là điểm nhấn portfolio rất tốt nếu trình bày đúng bản chất (transparent về scope và giới hạn dữ liệu).

**Key Concepts**
- Preference Data Collection (thủ công + bán tự động, không chờ user thật)
- Direct Preference Optimization (DPO) — offline
- Process vs Outcome Supervision (khái niệm, không nhất thiết implement cả hai)
- Reward Modeling cơ bản cho educational agent
- LoRA fine-tuning trên model nhỏ (Llama 3.2 1B/3B, Qwen2.5 nhỏ...)

**Coding Tasks (Tuần 10)**
☐ Xây pipeline thu thập preference data: tự tạo cặp response (tốt/xấu) từ interaction logs của Tuần 7-9 + tự đánh giá/chấm điểm (semi-synthetic, ghi rõ trong report)
☐ Chuẩn hóa dataset thành format preference pairs (chosen/rejected)
☐ Setup training environment (GPU cloud hoặc local nếu có)

**Coding Tasks (Tuần 11)**
☐ Training DPO/LoRA trên model nhỏ với dataset đã chuẩn bị
☐ Experiment tracking (WandB hoặc Comet)
☐ So sánh performance trước/sau fine-tuning bằng evaluation suite ở Tuần 7 (A/B so sánh, không cần A/B test với user thật)
☐ Viết rõ giới hạn: kích thước dataset, nguồn gốc data (synthetic vs real), scope của "cải thiện"

**Reading Materials**
- DPO paper, ORPO, RLAIF (đọc để hiểu khái niệm, không cần implement hết)
- "RL for LLM Agents" papers gần đây
- CMU AI Agents training section

**Evaluation Criteria**
- Agent có cải thiện đo được trên tập eval nhỏ (không cần "rõ rệt" ở quy mô lớn)
- Có báo cáo experiment trung thực: nguồn data, giới hạn, kết quả, và những gì sẽ cần để scale lên production thật

**⚠ Ghi chú quan trọng khi trình bày portfolio:** Gọi đúng tên đây là "proof-of-concept DPO fine-tuning" thay vì "RLHF hoàn chỉnh" hay "agent tự học online" — người phỏng vấn có kinh nghiệm sẽ hỏi sâu về nguồn data và quy mô, nên trung thực về scope sẽ đáng tin hơn là phóng đại.

# Tuần 12: Integration Buffer & Polish

**Tasks**
☐ Tích hợp lại module RL vào hệ thống chính (nếu áp dụng được) hoặc trình bày như module độc lập
☐ Dọn dẹp code, fix bug tồn đọng từ các tuần trước
☐ Chạy lại toàn bộ evaluation suite trên hệ thống hoàn chỉnh
☐ Buffer time cho bất kỳ phần nào bị trễ (thường là Tuần 4 hoặc Tuần 5-6)

# Tuần 13: Final Project & Portfolio

**Tasks**
☐ Hoàn thiện end-to-end system
☐ Viết technical blog / report (bao gồm phần RL với scope rõ ràng)
☐ Quay demo video
☐ Chuẩn bị portfolio + interview questions (đặc biệt chuẩn bị trả lời sâu về phần DPO: data, giới hạn, hướng scale)

**Deliverables**
- GitHub repo sạch sẽ
- Demo chạy được
- Technical write-up (kiểu blog), có phần riêng giải thích rõ ràng, trung thực về RL module

# Rủi ro & Ghi chú chung

| **Rủi ro** | **Mức độ** | **Giải pháp** |
| --- | --- | --- |
| Tuần 1 (PDF parsing) đội thời gian | Trung bình | Cắt table/image extraction thành stretch goal |
| Tuần 4 (Knowledge Graph) đội thời gian | Cao | Giới hạn cứng 5-6 ngày, hạ chuẩn entity types nếu cần |
| Tuần 5-6 (Q&A agent + kỹ thuật sư phạm) khó đo lường | Thấp-Trung bình *(giảm từ Trung bình-Cao sau khi định nghĩa lại vai trò reactive)* | Tiêu chí đo chất lượng phản hồi cụ thể, không đo "khả năng dẫn dắt chủ động" |
| Tuần 10-11 (RL/DPO): thiếu data thật | Cao | Dùng semi-synthetic data, ghi rõ giới hạn trong report |
| Tuần 10-11: cần GPU/compute | Trung bình | Dùng model nhỏ (1-3B) + LoRA, thuê GPU cloud ngắn hạn nếu cần |
| Phóng đại scope RL trong portfolio | Cao (rủi ro phỏng vấn) | Luôn trình bày trung thực là proof-of-concept |
| Không có buffer | Đã xử lý | Tuần 12 dành riêng làm buffer + tích hợp |

**Tổng thời gian: 13 tuần** (thay vì nén RL vào Tuần 8-9 cũ như bản gốc). Nếu part-time, có thể cần giãn thêm 1-2 tuần nữa.
