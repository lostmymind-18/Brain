# Bài học: Streamlit rerun xóa mọi nội dung đã render

**Ngữ cảnh:** Tuần 8-9 BookMind Tutor - fact-check expander hiện ngay sau khi
answer được stream xong, nhưng biến mất hoàn toàn sau khi `st.rerun()` được gọi.

---

## Vấn đề

Code ban đầu làm theo flow tự nhiên:

```python
# Trong _handle_question():
with st.chat_message("assistant"):
    # 1. Stream answer
    answer = st.write_stream(stream_generator())

    # 2. Run verification
    verification = verifier.verify(answer, chunks)

    # 3. Render fact-check expander
    _render_verification(verification)  # ✓ visible

# 4. Update session state
messages.append({"role": "assistant", "content": answer})

# 5. Rerun để update history
st.rerun()  # ← fact-check expander biến mất
```

Sau `st.rerun()`, Streamlit render lại page **từ đầu** bằng cách replay toàn bộ script.
Lúc đó nó render messages từ `st.session_state.messages` trong history loop - nhưng
verification result không được lưu vào messages, nên expander không được render lại.

---

## Root cause: Streamlit's execution model

Streamlit không có DOM, không có "persistent UI state".
Mỗi interaction (click, input, rerun) chạy lại toàn bộ Python script từ đầu.
Bất cứ thứ gì được render **chỉ tồn tại trong lần chạy đó**.

`st.session_state` là nơi duy nhất data survive qua reruns.

Hệ quả: nếu bạn render content trong một `if` block, một callback, hay một
temporary context manager - nó sẽ biến mất ngay khi rerun xảy ra trừ khi
bạn persist data vào `session_state`.

---

## Fix: Serialize verification vào message dict

Quy tắc: **chỉ render từ session_state, không render từ ephemeral local variables**.

```python
# Trong _handle_question() - serialize verification thành dict
verification_dict = None
if verification and verification.checks:
    verification_dict = {
        "checks": [
            {"claim": c.claim, "grounded": c.grounded, "source": c.source}
            for c in verification.checks
        ]
    }

# Lưu vào message dict (session_state persistent)
messages.append({
    "role": "assistant",
    "content": answer,
    "verification": verification_dict,  # ← key thêm
    ...
})

st.rerun()  # ← giờ OK, data đã trong session_state
```

```python
# Trong _chat_page() history loop - render từ message dict
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        _render_verification_dict(msg.get("verification"))  # ← từ session_state
```

---

## Pattern tổng quát

**Bất kỳ thứ gì cần survive qua rerun phải được stored trong `st.session_state`
trước khi `st.rerun()` được gọi.**

Điều này có nghĩa là: nếu logic của bạn tạo ra output sau khi user action, và output
đó cần appear lại trong lần render tiếp theo, bạn phải:

1. Compute output
2. Store vào `session_state` (thường là append vào list messages/history)
3. Gọi `st.rerun()`
4. Trong rerun: render từ `session_state`

Không bao giờ render trực tiếp từ local variable rồi rerun.

---

## Corollary: Serialization design

Khi serialize object vào session_state dict, không store Python object trực tiếp.
Store primitive types (dict, list, str, int, float, bool):

- Dễ debug (printable, JSON-serializable)
- Không có pickling issue
- Dễ migrate khi class thay đổi structure

Với `VerificationResult`:
```python
# Không làm: messages.append({..., "verification": verification_object})
# Làm: messages.append({..., "verification": {"checks": [...]}})
```

---

## Debugging hint

Khi Streamlit UI có nội dung bị "mất" sau interaction, check:
1. Có `st.rerun()` hoặc user trigger nào gây rerun không?
2. Nội dung bị mất có đang render từ local variable hay session_state?
3. Dùng `st.write(st.session_state)` để debug xem state có đúng không.
