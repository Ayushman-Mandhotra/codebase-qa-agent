from agents import ask

print("Codebase Q&A Agent — type a question, or 'quit' to exit.")
print("First question should mention the repo URL, e.g. 'Clone https://github.com/psf/requests and explain X'")
print("-" * 60)

thread_id = "cli-session"
while True:
    question = input("\nYou: ").strip()
    if question.lower() in {"quit", "exit"}:
        break
    answer = ask(question, thread_id=thread_id)
    print(f"\nAgent: {answer}")