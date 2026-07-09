# Bài học: Tại sao tự implement ReAct loop thay vì dùng LangGraph

**Ngữ cảnh:** Tuần 2 - xây `AgentHarness` với ReAct loop.

---

## Vấn đề ban đầu

Khi đọc roadmap Tuần 2, câu hỏi hiện ra ngay: *tại sao không dùng LangGraph?*
LangGraph là framework chính thống cho agent loop, được Anthropic và nhiều công ty lớn
khuyến nghị dùng trong production.
Dùng nó sẽ nhanh hơn nhiều, ít bug hơn, và code ngắn gọn hơn.

Vậy tại sao lại tự viết?

---

## Cái mà framework ẩn đi

Khi dùng LangGraph (hoặc LangChain AgentExecutor), framework xử lý:

- Build message list từ các turn
- Detect `stop_reason == "tool_use"` vs `"end_turn"`
- Extract content blocks từ response
- Format tool results thành đúng Anthropic format (`type: "tool_result"`, `tool_use_id`, `is_error`)
- Append assistant turn và tool_result turn vào đúng vị trí trong messages list
- Loop lại
- Handle max iterations

Tất cả những việc đó xảy ra trong vài dòng config.
Nhưng khi dùng framework, ta chỉ thấy "agent chạy" - không thấy *tại sao* nó chạy như vậy.

---

## Cái học được khi tự viết

Tự implement buộc phải đối mặt trực tiếp với những chi tiết quan trọng:

**1. Messages list có cấu trúc bất biến**

Anthropic API yêu cầu messages list phải alternating user/assistant.
`tool_result` phải nằm trong một `user` turn - không phải turn riêng.
Khi tool_use xảy ra, assistant turn phải được append *trước* khi append tool results.
Nếu sai thứ tự, API trả về lỗi 400.

Khi tự viết `ConversationMemory.add_assistant()` rồi `add_tool_results()`, thứ tự này
trở nên rõ ràng và chủ động - không phải magic ẩn trong framework.

**2. "Tool use" chỉ là một stop_reason**

ReAct loop về bản chất là:

```
loop:
    response = call_claude(messages, tools)
    if response.stop_reason == "end_turn":
        return response.text
    if response.stop_reason == "tool_use":
        results = execute_tools(response.content)
        messages += [assistant_turn, tool_result_turn]
        continue
```

Đơn giản đến mức ngạc nhiên.
Không có magic.
Không có graph.
Không có state machine.
Chỉ là một vòng lặp while với hai nhánh if.

**3. "Reasoning" trong ReAct không cần prompt đặc biệt**

Paper ReAct gốc (Yao et al. 2022) dùng các model GPT-3 cũ, cần prompt engineer
thêm "Thought: ... Action: ... Observation: ..." để model tự viết reasoning chain.

Với Claude, điều này không cần thiết.
Claude *tự nhiên* suy nghĩ trước khi gọi tool - reasoning ẩn trong quá trình generation.
Framework nào đó cố thêm "Thought" block vào prompt thực ra đang làm dư thừa với Claude hiện tại,
thậm chí có thể làm giảm chất lượng nếu constrain sai.

**4. Error handling là trách nhiệm của người viết harness**

Khi tool fail, framework thường có một behavior mặc định.
Tự viết buộc phải quyết định: retry? return error? crash?
Quyết định: return `is_error=True` để Claude tiếp tục loop thay vì crash,
với retry chỉ cho transient errors (network, timeout) - không retry logic errors.

Đây là design decision thực sự, không phải config một option.

---

## Khi nào nên dùng framework

Sau khi hiểu cơ chế, dùng LangGraph (hay tương đương) là hợp lý khi:

- Cần production-grade observability (tracing, replay)
- Cần parallel tool execution
- Cần human-in-the-loop interruption
- Cần persistent state qua nhiều sessions (checkpoint)
- Team size lớn, cần standardization

Dùng thủ công là hợp lý khi:
- Đang học (mục tiêu của Tuần 2)
- Control flow đơn giản và không thay đổi
- Muốn zero dependency overhead

---

## Takeaway

> **Hiểu cơ chế trước, dùng abstraction sau.**
>
> Framework tốt nhất là framework mà bạn biết nó đang làm gì bên trong.
> Viết tay một lần, rồi dùng framework với sự tự tin - không phải với sự mù quáng.

Tuần 3 sẽ implement các reasoning strategies khác nhau (Plan-and-Execute, Reflexion)
trên cùng `AgentHarness` này.
Khi đó sẽ thấy rõ hơn: harness là infrastructure cố định, strategy là plugin thay được.
Một lesson khác cho lúc đó.
