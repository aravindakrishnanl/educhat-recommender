from chat import edu_chat

def test_edu_chat_education_question(query):
    response = edu_chat(query)

    return response

print(test_edu_chat_education_question("What are some effective study techniques for learning a new language?"))


