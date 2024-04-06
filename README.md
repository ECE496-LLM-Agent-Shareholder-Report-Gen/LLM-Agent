# LLM Agent for Shareholder/Regulator Report Generation

Capstone project for ECE496 at the University of Toronto.

## Created by:
- Sebastion Czyrny (sebastian.czyrny@mail.utoronto.ca)
- Danny Ahmad (danial.ahmad@mail.utoronto.ca)
- David Marcovitch (david.marcovitch@mail.utoronto.ca)
- Mert Okten (mert.okten@mail.utoronto.ca)

## Table of Contents

- [Installation](#installation)
- [Contributing](#contributing)
- [License](#license)


## Installation

1. Ensure you have Python 3.x installed on your system.
2. Clone the repository to your local machine:
   ```console
   git clone https://github.com/ECE496-LLM-Agent-Shareholder-Report-Gen/LLM-Agent.git
   ```
3. Navigate to the project directory.
4. (Optional) Create and activate a virtual environment to isolate your project dependencies:
   ```console
   python3 -m venv venv
   source venv/bin/activate # On Linux/Mac
   .\venv\Scripts\activate # On Windows
   ```
5. Install the required dependencies using pip:
   ```console
   pip install -r requirements.txt
   ```
6. Set your Ollama environment variable:
   ```console
   export OLLAMA_MODELS=/groups/acmogrp/Large-Language-Model-Agent/language_models/ollama
   ```
7. Navigate to ollama folder and run Ollama server:
   ```console
   ./ollama-linux-amd64 serve&
   ```
8. Run streamlit app:
   ```console
   streamlit run streamlit_app.py
   ```


## Contributing

Thank you for considering contributing to our project! Contributions are welcome from everyone.

To contribute to this project, please follow these guidelines:

1. Fork the repository and clone it to your local machine.
2. Create a new branch for your contribution: `git checkout -b feature/new-feature`.
3. Make your changes and test them thoroughly.
4. Commit your changes: `git commit -m "Add new feature"`.
5. Push to your branch: `git push origin feature/new-feature`.
6. Submit a pull request, describing your changes in detail and mentioning any related issues.
7. After submitting the pull request, our team will review your changes and provide feedback as needed.

Please ensure that your contributions adhere to our [code of conduct](CODE_OF_CONDUCT.md).

If you have any questions or need assistance with the contribution process, feel free to reach out to us by creating an issue.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
