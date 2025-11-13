gunicorn app.main:app -b 0.0.0.0:4545 -w 1 -k uvicorn.workers.UvicornWorker --timeout 120

With Session

{
  "session_id": "jee_physics01",
  "user_id": "student_123",
  "assistant_id": "asfasfasf",
  "messages": [
    {
      "role": "user",
      "content": "what are the NEWTON's laws?"
    }
  ],
  "stream": false,
  "top_k": 5,
  "temperature": 0.3
}

---

{
  "session_id": "jee_physics01",
  "user_id": "student_123",
  "assistant_id": "asfasfasf",
  "messages": [
    {
      "role": "user",
      "content": "Two point charges, $+Q$ and $-Q$, are placed a distance $d$ apart. What is the electric field magnitude at a point exactly midway between them?"
    }
  ],
  "stream": false,
  "top_k": 5,
  "temperature": 0.3
}

---

{
  "session_id": "jee_physics01",
  "user_id": "student_123",
  "assistant_id": "asfasfasf",
  "messages": [
    {
      "role": "user",
      "content": "State the key observation that demonstrates the particle nature of light in the photoelectric effect, and write the Einstein's photoelectric equation."
    }
  ],
  "stream": false,
  "top_k": 5,
  "temperature": 0.3
}

---

{
  "session_id": "jee_physics01",
  "user_id": "student_123",
  "assistant_id": "asfasfasf",
  "messages": [
    {
      "role": "user",
     "content": "A steel rod is fixed between two rigid supports at $20^{\\circ}\\text{C}$. The rod is then heated to $50^{\\circ}\\text{C}$. If the Young's modulus of steel is $Y$ and its coefficient of linear expansion is $\\alpha$, derive the expression for the thermal stress developed in the rod."
    }
  ],
  "stream": false,
  "top_k": 5,
  "temperature": 0.3
}

---

# Search

{
    "query": "Newtons laws of motion",
    "top_k": 5
}

---

# Chat

{
  "messages": [
    {
      "role": "user",
      "content": "Explain Newton's laws of motion with examples"
    }
  ],
  "stream": false,
  "subject_filter": null,
  "topic_filter": null,
  "top_k": 5,
  "temperature": 0.7
}

{
  "messages": [
    {
      "role": "user",
      "content": "Explain Newton's laws of motion with examples"
    }
  ],
  "stream": false,
  "subject_filter": null,
  "topic_filter": null,
  "top_k": 5,
  "temperature": 0.7
}

{
  "messages": [
    {
      "role": "user",
      "content": "Explain chemical bonding and molecular orbital theory in detail"
    }
  ],
  "stream": true,
  "subject_filter": "Chemistry",
  "topic_filter": "ChemicalBonding",
  "top_k": 5,
  "temperature": 0.7
}
