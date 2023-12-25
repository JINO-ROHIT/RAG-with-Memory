# Rag with Memory 📖

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Description

Rag with Memory is a project that leverages Llama 2 7b chat assistant to perform RAG (Retrieval-Augmented Generation) on uploaded documents. Additionally, it operates in a chat-based setting with short-term memory by summarizing all previous K conversations into a standalone conversation to build upon the memory.
Inspired from langchain - https://python.langchain.com/docs/use_cases/question_answering/#adding-memory

The database used is from the amazing Vlite repository which builds on numpy - https://github.com/sdan/vlite

The user interface (UI) provides the ability to change the current prompt and experiment with the most important text generation parameters. The little left corner of the UI keeps track of the tokens generated over the session.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Installation

To get started with Rag with Memory, follow these steps:

```bash
git clone https://github.com/your-username/rag-with-memory.git
cd rag-with-memory
streamlit run app.py
```

1. Open the application in your web browser.
2. Upload a document for RAG processing.
3. Explore the chat-based setting with short-term memory.
4. Use the UI to experiment with different prompts and text generation parameters.

## Features
1. Regular old RAG on the given document using a numpy database(Vlite)
2. Chat-based setting with short-term memory
3. UI for changing prompts and experimenting with text generation parameters
4. Token count tracking in the UI


## Contributing
If you'd like to contribute to Rag with Memory, follow these steps:

Fork the repository.

Create a new branch (git checkout -b feature/new-feature).

Make your changes and commit them (git commit -am 'Add new feature').

Push to the branch (git push origin feature/new-feature).

Create a new pull request.


## License
This project is licensed under the MIT License - see the LICENSE file for details.

### Contact
For any questions or feedback, feel free to reach out:

Email: find.jinorohit@gmail.com
