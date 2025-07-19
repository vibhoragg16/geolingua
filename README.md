# GeoLingua

**A** geographic-aware language model that adapts its **responses based on regional linguistic patterns, cultural references, and local knowledge using Group-based Reward Policy Optimization (GRPO) techniques.**

## 📝 Overview

GeoLingua is an open-source project designed to understand and generate text appropriate to specific geographic regions. It moves beyond generic language models by incorporating local dialects, cultural references, and regional knowledge. The model uses **Group-based Reward Policy Optimization (GRPO)** to fine-tune responses for different geographic contexts, ensuring more natural and contextually-aware interactions.

## ✨ Features

* **Geographic Adaptation:** Adapts language style based on the target geographic context.
* **Cultural Awareness:** Incorporates regional cultural references, slang, and local knowledge.
* **GRPO Training:** Uses Group-based Reward Policy Optimization for fine-tuning, a novel approach to steer the model's behavior.
* **Multi-source Data Collection:** Gathers data from diverse sources like Reddit, news articles, and Wikipedia to build robust regional profiles.
* **Supported Regions:** Pre-trained models for a variety of English-speaking regions.
* **Interactive Demo:** A web-based interface built with Streamlit for easy testing and demonstration.
* **Extensible:** Designed to be easily extended with new regions and data sources.

## 🌏 Supported Regions

* **United States (South):** Southern dialects, cultural references, and local knowledge.
* **United Kingdom:** British English, cultural context, and regional variations.
* **Australia:** Australian English, slang, and cultural references.
* **India:** Indian English variations and cultural context.
* **Nigeria:** Nigerian English and cultural references.

## 🏗️ Project Structure

```
geolingua/
├── src/
│   ├── data/          # Data collection and preprocessing scripts
│   ├── models/        # Model architecture and training logic
│   ├── evaluation/    # Evaluation metrics and benchmarks
│   └── utils/         # Utility functions
├── config/            # Configuration files for models and data
├── data/              # Raw and processed data storage
├── notebooks/         # Jupyter notebooks for experimentation
├── demo/              # Interactive Streamlit demo application
├── tests/             # Unit and integration tests
└── scripts/           # High-level scripts for training, evaluation, etc.
```

## ⚙️ Installation

To get a local copy up and running, follow these steps.

### Prerequisites

* Python 3.8+
* pip (Python package installer)
* Git

### Steps

1. **Clone the repository:**
   ```sh
   git clone [https://github.com/vibhoragg16/geolingua.git](https://github.com/vibhoragg16/geolingua.git)
   cd geolingua
   ```

2. **Create and activate a virtual environment:**
   * **Linux/macOS:**
     ```sh
     python3 -m venv geolingua_env
     source geolingua_env/bin/activate
     ```
   * **Windows:**
     ```sh
     python -m venv geolingua_env
     geolingua_env\Scripts\activate
     ```

3. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

4. **Install in development mode (recommended for contributors):**
   ```sh
   pip install -e .
   ```

## 🚀 Quick Start

Follow these steps to collect data, train a model, and run the demo.

1. **Data Collection:**
   Collect data for a specific region from a source like Reddit.
   ```sh
   python scripts/collect_data.py --region us_south --source reddit
   ```

2. **Training:** Train the model using kaggle and for more detail refer KAGGLE_GUIDE.md
   ```sh
   python kaggle_train.py
   ```

3. **Evaluation:**
   Evaluate the performance of your trained model.
   ```sh
   python evaluate_model.py
   ```

4. **Run the Interactive Demo:**
   Launch the Streamlit demo to interact with your model.
   ```sh
   streamlit run demo/app.py
   ```

## 🔧 Configuration

Model, training, and data parameters can be easily modified in the `config/` directory:

* `config/model_config.py`: Controls model architecture, learning rates, and other training settings.
* `config/data_config.py`: Manages data collection sources, preprocessing steps, and paths.

## 🤝 Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

Please add tests for any new features or bug fixes.

## 📜 License

Distributed under the MIT License. See `LICENSE` file for more information.

## ✒️ Citation

If you use GeoLingua in your research, please cite it as follows:

```bibtex
@software{geolingua_2025,
  title={GeoLingua: A Geographic-Aware Language Model with Group-based Reward Policy Optimization (GRPO)},
  author={Vibhor Aggarwal},
  year={2025},
  url={[https://github.com/vibhoragg16/geolingua](https://github.com/vibhoragg16/geolingua)}
}
```

## 📧 Contact

Vibhor Aggarwal - vibhoragg16@email.com

Project Link: <https://github.com/vibhoragg16/geolingua>
