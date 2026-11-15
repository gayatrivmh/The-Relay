 The Relay – Reliable RAG-Based Question Answering System

 Overview

The Relay is a Retrieval-Augmented Generation (RAG) based document question-answering system built using Python and Groq's LLM API.

It answers questions using information retrieved from source documents. The system focuses on generating evidence-supported answers, providing page-level citations, and refusing questions when sufficient information is unavailable.

The project also includes reliability experiments to study how AI workflows behave under API rate limits, token limits, timeouts, and high request volumes.

 Key Features

- Query rewriting to improve question retrieval.
- Hybrid retrieval using BM25 and vector search.
- Retrieval of the top 3 relevant document chunks.
- Relevance checking before answer generation.
- Answer generation with page-level citations.
- Verification of generated answers against retrieved evidence.
- Evidence-based refusal for questions without sufficient information.
- API reliability and performance testing.

 System Workflow

The system follows a five-stage pipeline:

1. Query Rewriting: Rewrites the original question to improve retrieval.
2. Hybrid Retrieval: Retrieves relevant document chunks using BM25 and vector search.
3. Relevance Checking: Evaluates whether the retrieved content is useful for answering the question.
4. Answer Generation: Generates an answer using the retrieved evidence and includes page-level citations.
5. Answer Verification: Checks whether the generated answer is supported by the source material and refuses unsupported answers.

 Tech Stack

- Programming Language: Python
- LLM: Groq API – GPT-OSS-20B
- Retrieval: BM25 and vector search
- Embeddings: FastEmbed
- API Framework: FastAPI
- Libraries: NumPy, HTTPX, Rank-BM25

 Reliability and Evaluation

A major part of this project involved evaluating answer quality and system reliability.

The evaluation considered:

- Answer correctness
- Retrieval relevance
- Citation quality
- Refusal accuracy
- Response latency and p95 latency
- Successful request rate
- API errors, rate limits, and timeouts

A FastAPI-based API simulator was developed to reproduce API constraints and test different traffic conditions.

Experiments explored:
- Steady, burst, and overload traffic.
- Rate-limit and token-limit behaviour.
- Retry strategies.
- Request gating and token gating.
- Latency and failure rates under different workloads.

 Repository Structure

  pipeline.py – Main question-answering pipeline.

  retrieval.py – Document retrieval functionality.

  fake_groq.py – API behaviour simulator.

  gate.py – Request gating logic.

  tokengate.py – Token-based request gating.

  load_test.py – Load testing.

  retry_experiment.py – Retry experiments.

  gate_experiment.py – Request gating experiments.

  tokengate_experiment.py – Token gating experiments.

  run_questions.py – Question evaluation runner.

  results.csv – Experimental results.

  questions.txt – Evaluation questions.

  requirements.txt – Python dependencies.

 Setup

 1. Clone the repository

git clone YOUR_GITHUB_REPOSITORY_URL
cd The-Relay

 2. Install dependencies

pip install -r requirements.txt

 3. Configure the Groq API key

Set your API key as an environment variable.

For Windows PowerShell:

$env:GROQ_API_KEY="your_groq_api_key"

Never commit your actual API key or `.env` file to GitHub.

 Limitations

- The system depends on the quality and relevance of the source documents.
- Retrieval and answer quality can vary depending on the question and retrieved context.
- API rate limits and network conditions can affect response time and request success.
- Reliability experiments use a simulated API environment and do not represent a full production deployment.
- The project does not implement persistent conversational memory.

 Learning Outcomes

Through this project, I gained practical experience in:

- Building a multi-stage RAG workflow.
- Integrating an existing LLM through an API.
- Implementing hybrid information retrieval.
- Evaluating AI-generated answers and citations.
- Handling API constraints and failures.
- Measuring latency, reliability, and system performance.

 Author

Gayatri Vijay Mhamane

B.E. – Artificial Intelligence and Data Science

Pune, Maharashtra, India