from model import medical_safety_assessment

print("Medical Safety Checker")
print("Type 'exit' to quit\n")

while True:
    user_input = input("Enter text: ")

    if user_input.lower() == "exit":
        break

    result = medical_safety_assessment(user_input)

    print("\n--- Result ---")
    for k, v in result.items():
        print(f"{k}: {v}")
    print()