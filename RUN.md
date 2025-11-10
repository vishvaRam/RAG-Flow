gunicorn app.main:app -b 0.0.0.0:4545 -w 1 -k uvicorn.workers.UvicornWorker --timeout 120

### With Session

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
  "session_id": "jee_physics_session_001",
  "user_id": "student_12345",
  "assistant_id": "rag_assistant_v1",
  "messages": [
    {
      "role": "user",
      "content": "A uniform disc of mass M and radius R is rotating with angular velocity ω about an axis passing through its center and perpendicular to its plane. A particle of mass m (where m << M) is gently placed on the disc at a distance R/2 from the center. The coefficient of friction between the particle and disc is μ. Find the time taken by the particle to start rotating with the same angular velocity as the disc, and determine the angular momentum of the system after the particle attains the angular velocity of the disc."
    }
  ],
  "stream": false,
  "top_k": 5,
  "temperature": 0.3
}

---

{
  "session_id": "jee_chemistry_session_002",
  "user_id": "student_12345",
  "assistant_id": "rag_assistant_v1",
  "messages": [
    {
      "role": "user",
      "content": "Consider the complex [Co(NH3)4Cl2]+ which exhibits geometric isomerism. (a) Draw all possible isomers and indicate which are optically active. (b) If this complex undergoes a ligand substitution reaction where one NH3 is replaced by H2O, how many isomers of the product [Co(NH3)3(H2O)Cl2]+ are possible? (c) Using Crystal Field Theory, explain the electronic configuration and calculate the Crystal Field Stabilization Energy (CFSE) for each isomer, assuming Δ0 > P (pairing energy)."
    }
  ],
  "stream": false,
  "top_k": 7,
  "temperature": 0.2
}

---

{
  "session_id": "jee_maths_session_003",
  "user_id": "student_12345",
  "assistant_id": "rag_assistant_v1",
  "messages": [
    {
      "role": "user",
      "content": "Let z1, z2, z3, ..., z10 be 10 distinct complex numbers lying on the unit circle |z| = 1 in the complex plane. Prove that |z2² - z1²| + |z3² - z2²| + ... + |z10² - z9²| + |z1² - z10²| ≤ 4π. Also, find the condition for equality to hold. Additionally, if ω is a complex cube root of unity, evaluate: Σ(k=1 to n) k·(k+1-ω)·(k+1-ω²)"
    }
  ],
  "stream": true,
  "top_k": 6,
  "temperature": 0.25
}

---

{
  "session_id": "jee_physics_thermo_005",
  "user_id": "student_99999",
  "assistant_id": "rag_assistant_v1",
  "messages": [
    {
      "role": "user",
      "content": "Three very large plates of the same area are kept parallel and close to each other. They are considered as ideal black surfaces and have very high thermal conductivity. The first and third plates are maintained at temperatures 2T and 3T respectively. Find: (a) the temperature of the middle plate under steady state condition, (b) the rate of heat transfer per unit area from plate 1 to plate 2, and (c) the net energy radiated per unit area from the middle plate. Use Stefan-Boltzmann law where σ is the Stefan-Boltzmann constant."
    }
  ],
  "stream": false,
  "top_k": 5,
  "temperature": 0.2
}

---

{
  "session_id": "jee_maths_calculus_006",
  "user_id": "student_11111",
  "assistant_id": "rag_assistant_v1",
  "messages": [
    {
      "role": "user",
      "content": "Let g: [0, π/2] → ℝ be a continuous function defined by g(x) = ∫[0 to x] (sin(t)/(1 + t²)) dt. Consider the area bounded by the curve y = g(x), the x-axis, and the lines x = 0 and x = π/2. (a) Prove that g(x) is an increasing function on the given interval. (b) Find g'(π/4). (c) Calculate the approximate area under the curve using Simpson's rule with 4 subintervals. (d) If the curve is rotated about the x-axis, find the volume of the solid of revolution."
    }
  ],
  "stream": false,
  "top_k": 6,
  "temperature": 0.25
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
