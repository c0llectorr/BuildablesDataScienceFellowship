# My Data Science Fellowship Journey at Buildables

A comprehensive account of my 12-week remote fellowship as a Data Science Intern, documenting the progression from foundational programming to production-grade deep learning systems.

---

## Table of Contents

1. [Overview](#overview)
2. [Fellowship Structure](#fellowship-structure)
3. [Week-by-Week Breakdown](#week-by-week-breakdown)
4. [Individual Projects](#individual-projects)
5. [Peer Collaboration](#peer-collaboration)
6. [Hackathon Experience](#hackathon-experience)
7. [Final Capstone Project](#final-capstone-project)
8. [Key Technologies](#key-technologies)
9. [Datasets & Models](#datasets--models)
10. [Reflections & Growth](#reflections--growth)

---

## Overview

The Buildables Data Science Fellowship was a transformative 12-week remote program that took me from learning Python fundamentals to deploying production-ready machine learning systems. This repository documents every project, assignment, and learning milestone from that journey.

What started as nervous excitement in Week 1 evolved into genuine confidence by Week 12. I moved through the complete machine learning lifecycle: data engineering, statistical analysis, supervised and unsupervised learning, deep learning, natural language processing, and full-stack deployment. More importantly, I learned how to think with data, solve real problems, accept feedback, and grow through consistent effort.

The fellowship wasn't just about technical skills. It was about learning to collaborate, lead teams, manage time under pressure, and translate complex technical work into actionable insights. It was about understanding that data science is as much art as it is science.

---

## Fellowship Structure

The 12-week program was organized into distinct phases, each building on the previous:

- **Weeks 1-4:** Foundations and core concepts
- **Weeks 5-6:** First peer project with real-world application
- **Week 7:** Hackathon participation
- **Week 8:** Deep learning theory
- **Weeks 9-10:** Unsupervised learning and daily tasks
- **Weeks 9-12:** Capstone project (overlapping with other work)

This structure allowed for both breadth and depth, ensuring I understood fundamentals while also having time to build sophisticated systems.

---

## Week-by-Week Breakdown

### Week 1: Getting Started with Foundations

The first week was about establishing a strong foundation. I worked through four comprehensive tasks that covered the essential tools and concepts needed for the rest of the fellowship.

**Task 1: Python Fundamentals**
Nine programming problems that built logical reasoning and Python syntax fluency. These weren't trivial exercises—they required thinking about edge cases, optimization, and clean code. Problems included Fibonacci sequences, palindrome checking, string reversal, and mathematical operations. Coming from a C++ background, I was impressed by Python's expressiveness and readability.

**Task 2: Object-Oriented Programming & Data Structures**
This task reinforced OOP principles like classes, inheritance, and encapsulation. I implemented data structures and algorithms, understanding not just how they work but why they matter. The focus on algorithmic complexity helped me think about scalability from day one.

**Task 3: Advanced Data Structures & Algorithms**
Binary search trees, linked lists, dynamic programming, and interval merging. These problems required deeper algorithmic thinking and helped me appreciate the elegance of well-designed solutions.

**Task 4: Data Science Fundamentals**
The most exciting part of Week 1. I loaded datasets using scikit-learn, manipulated data with Pandas, created visualizations with Matplotlib, and trained my first machine learning model. Working with the Iris, Breast Cancer, and California Housing datasets made abstract concepts concrete. I remember the moment I trained a Linear Regression model and saw it actually predict values—it felt like magic, but it was just math.

**Key Insight:** The transition from pure programming to data science felt natural because the foundational thinking was the same: break problems into smaller pieces, understand the data, and iterate.

---

### Week 2: Statistics, Probability, and Data Preprocessing

Week 2 shifted focus to the statistical foundations that underpin all data science work.

**Task 1: Descriptive and Inferential Statistics**
I worked through central tendencies, dispersion measures, and summary statistics. Then moved into hypothesis testing and probability distributions. These concepts felt abstract until I started applying them to real datasets.

**Task 2: Data Manipulation at Scale**
Twenty tasks using Pandas and NumPy. This was where I truly learned data manipulation—not just the syntax, but the thinking behind it. How do you handle missing values? When do you drop rows versus impute? How do you structure data for analysis?

**Task 3: Feature Engineering and Data Cleaning**
Working with Boston Housing, Titanic, and Car Evaluation datasets, I learned practical techniques: feature scaling, KNN imputation, handling categorical variables, and dealing with missing data. The Titanic dataset was particularly instructive—it's messy, it's real, and it taught me that data science is often 80% cleaning and 20% modeling.

**Key Insight:** Data quality determines model quality. No amount of sophisticated algorithms can compensate for poor data preparation.

---

### Week 3: Real-World Analytics—Feedback Dashboard Project

Week 3 was my first solo project that felt like real work. I was tasked with analyzing fellowship feedback data and creating a dashboard with actionable insights.

**The Project:**
I received raw feedback data from fellows across the program. My job was to clean it, analyze it, and create a dashboard that program managers could use to improve the fellowship.

**What I Did:**
1. Cleaned the data in Excel, handling inconsistencies and formatting issues
2. Created an interactive dashboard in Monday.com with key metrics
3. Wrote a 6-page report with insights and recommendations

**Key Findings:**
- Geographic imbalance: 11 applicants from Pakistan versus 1 from Nigeria
- High daily learning effectiveness across the board
- Opportunities for deeper peer collaboration
- Specific recommendations for program improvement

**Why This Mattered:**
This project taught me that data science isn't just about building models. It's about understanding context, asking the right questions, and translating technical findings into business language. The dashboard had to be intuitive for non-technical stakeholders. The report had to be concise but comprehensive.

**Key Insight:** The best analysis is useless if it doesn't lead to action. Communication is as important as computation.

---

### Week 4: Supervised Learning—Regression and Classification

Week 4 deepened my understanding of supervised learning through four focused tasks.

**Task 1: Linear Regression**
Predicting student performance using regression. I learned about coefficients, intercepts, and how to interpret model outputs.

**Task 2: Classification Fundamentals**
Working with the Diabetes dataset, I built classification models and learned about accuracy, precision, recall, and F1 scores. These metrics aren't just numbers—they tell you different stories about model performance.

**Task 3: Multi-Algorithm Classification**
Comparing multiple algorithms (Decision Trees, Random Forests, SVM) on different datasets. This taught me that there's no one-size-fits-all solution. Different algorithms have different strengths.

**Task 4: Model Evaluation**
Deep dive into evaluation techniques using the Titanic dataset. Cross-validation, confusion matrices, ROC curves—these tools help you understand not just how well your model performs, but where it fails.

**Key Insight:** Model building is iterative. You train, evaluate, understand failures, and improve. The first model is rarely the best model.

---

### Week 5-6: Peer Project—Food Delivery Delay Prediction

This was my first major collaborative project, and it was transformative. I worked with two teammates to build a complete machine learning system from data to deployment.

**The Team:**
- Muhammad Ahmad (me): Project Lead, Data Science
- Hira Arif: Backend Development
- Abdul Basit: Data Analysis (had to leave due to emergency)

**The Problem:**
Food delivery services lose money on late deliveries and customer satisfaction suffers. Can we predict which deliveries will be late and why?

**The Data:**
Approximately 1,000 delivery records with features like distance, weather, traffic level, vehicle type, preparation time, and courier experience. The data was real but messy—missing values, inconsistent formatting, outliers.

**The Pipeline:**

1. **Data Exploration & Cleaning**
   - Identified 30 missing values per feature
   - Analyzed distributions and relationships
   - Handled missing data through appropriate imputation

2. **Feature Engineering**
   - Created base delivery time calculation
   - Incorporated traffic and weather factors
   - Adjusted for courier experience
   - Engineered target variable (Is_Late)

3. **Model Development**
   - **Regression Model:** XGBoost Regressor to predict actual delivery time
     - R² Score: 0.81 (explains 81% of variance)
     - Mean Squared Error: 85.25
   - **Classification Model:** XGBoost Classifier to predict late vs. on-time
     - Accuracy: 94.5%
     - Precision: 95.9%
     - Recall: 93.0%
     - F1 Score: 94.4%
     - ROC AUC: 94.5%

4. **Deployment**
   - Built Flask web application for real-time predictions
   - Created Power BI dashboard for stakeholder visualization
   - Integrated both models into a single system

**Key Findings:**
- Distance is the most influential factor in delivery time
- Traffic conditions significantly impact delays
- Weather has moderate effect, especially in extreme conditions
- Courier experience shows diminishing returns after 3-4 years
- Bikes are fastest in urban areas

**Business Impact:**
The system could improve on-time delivery by 15-20% through better resource allocation and more accurate customer expectations.

**What I Learned:**
- How to lead a technical team
- How to manage scope when a team member leaves
- How to balance model complexity with interpretability
- How to communicate technical results to non-technical stakeholders
- That real-world data is messier than textbook examples

**Key Insight:** The best model is the one that gets used. We built something that worked, was understandable, and could be deployed quickly.

---

### Week 6: Daily Task—Book Recommendation System

While working on the food delivery project, I also completed a daily task on recommendation systems using Microsoft Fabric.

**The Project:**
Build and evaluate a book recommendation system using collaborative filtering techniques.

**Platform:** Microsoft Fabric LakeHouse environment

**What I Built:**
- Data pipeline for book and user data
- Recommendation algorithm implementation
- Evaluation metrics and scoring system

**Key Insight:** Recommendation systems are everywhere, and understanding how they work is crucial for modern data science.

---

### Week 7: Hackathon—ExoVision AI (NASA Space Apps 2025)

Week 7 was unlike anything else in the fellowship. I participated in the NASA Space Apps Hackathon 2025 with a team of five passionate people.

**The Team:**
- Muhammad Ahmad (me): Team Lead & Data Scientist
- Syed Darain Hyder Kazmi: ML Engineer
- Muhammad Ahsan Atiq: Backend Developer
- Muhammad Mohsin: Frontend Developer
- Ali Hassan: Research Lead

**The Challenge:**
Use NASA's open data to solve a real-world problem. We chose exoplanet detection.

**The Problem:**
NASA has discovered thousands of exoplanets, but many more candidates await classification. Can we use machine learning to automatically classify celestial objects as confirmed exoplanets or planetary candidates?

**The Solution:**

1. **Data Source:** NASA K2 Planets and Candidates Catalog

2. **ML Pipeline:**
   - Preprocessing and feature engineering on stellar and orbital parameters
   - Parallel model development (two sub-teams working on different algorithms)
   - Team A: XGBoost and Decision Tree models
   - Team B: LightGBM model
   - 5-fold stratified cross-validation
   - Model selection based on performance comparison

3. **Model Performance:**
   - Decision Tree: ~99% cross-validation accuracy
   - XGBoost: Selected for superior performance and interpretability
   - LightGBM: Used for benchmarking

4. **Web Application (Flask):**
   - CSV upload mode for batch classification
   - Manual input mode for real-time predictions
   - Dynamic visualizations (distributions, heatmaps, ROC curves)
   - Side-by-side model comparison

**The Experience:**
Hackathons are intense. We had limited time, high stakes, and the need to make quick decisions. I learned that sometimes "good enough" is better than "perfect," and that communication within a team under pressure is critical. We had to divide work efficiently, trust each other's expertise, and come together at the end with a cohesive product.

**Key Insight:** Hackathons teach you how to work under pressure and make decisions with incomplete information. These are real-world skills.

---

### Week 8: Deep Learning Theory

Week 8 was dedicated to understanding deep learning fundamentals. I worked through neural network architectures, backpropagation, optimization algorithms, and activation functions.

**What I Studied:**
- Perceptrons and multi-layer networks
- Backpropagation and gradient descent
- Optimization algorithms (SGD, Adam, RMSprop)
- Activation functions and their properties
- Regularization techniques (dropout, batch normalization)
- Convolutional and recurrent architectures

**Key Insight:** Deep learning is powerful, but it's not magic. Understanding the math behind it helps you debug models and make informed architectural choices.

---

### Week 9: Daily Task—Iris Clustering Analysis

A focused task on unsupervised learning using the classic Iris dataset.

**What I Did:**
- Applied K-Means clustering
- Evaluated cluster quality using silhouette analysis
- Visualized clusters in 2D and 3D
- Compared different numbers of clusters

**Key Insight:** Unsupervised learning is harder to evaluate than supervised learning because there's no ground truth. You have to think creatively about what "good" clustering means.

---

### Week 10: Daily Task—Cluster Analysis Comparison

Extending the clustering work, I compared K-Means and Hierarchical clustering on multiple datasets.

**Datasets:** Iris and Mall Customers

**What I Compared:**
- Cluster quality metrics
- Computational efficiency
- Interpretability
- Sensitivity to parameters

**Key Insight:** Different clustering algorithms have different strengths. K-Means is fast but assumes spherical clusters. Hierarchical clustering is more flexible but slower.

---

### Week 9-12: Final Capstone Project—MoodFlix AI

The final project was the culmination of everything I learned. It's a production-grade deep learning system that detects emotions from text and recommends movies based on mood.

**The Vision:**
What if you could tell a system how you're feeling, and it would recommend movies that match your emotional state? Not just any movies, but movies that align with your current mood.

**The Architecture:**

A three-tier system combining data engineering, deep learning, and full-stack web development.

**Data Layer: SuperEmotion Dataset**

- Original dataset: 552,821 samples across 7 emotions
- Cleaning pipeline:
  - Filtered to 7 target emotions (anger, fear, joy, love, neutral, sadness, surprise)
  - Applied class balancing (capped majority classes at 30,000 samples)
  - Stratified train/validation/test split (80/10/10)
- Final dataset: 190,716 samples with improved class balance

**Model Layer: DeBERTa v3 Base**

DeBERTa (Decoding-enhanced BERT with disentangled attention) is a state-of-the-art transformer model. It improves on BERT through:
- Disentangled attention (separates content and position embeddings)
- Enhanced mask decoder
- Relative position encoding

Model specifications:
- 183 million parameters
- 12 transformer layers
- 12 attention heads
- 768 hidden dimensions

Training configuration:
- Optimizer: AdamW (learning rate: 2e-5)
- Batch size: 32
- Epochs: 3 with early stopping
- Mixed precision (FP16) for efficiency
- Class weights to handle imbalance

**Results:**
- Test Accuracy: 90.54%
- F1 Macro: 89.92%
- AUC-ROC Macro: 99.37%

Per-class performance:
- Sadness: 94.09% F1
- Joy: 94.01% F1
- Anger: 91.95% F1
- Love: 92.61% F1
- Fear: 90.45% F1
- Neutral: 84.02% F1
- Surprise: 82.33% F1

The model generalizes exceptionally well—validation and test performance differ by less than 1%, indicating no overfitting.

**Application Layer: Full-Stack Deployment**

Backend (FastAPI on HuggingFace Spaces):
- Async request handling for concurrent users
- Endpoints for health checks, emotion prediction, batch processing, and recommendations
- TMDB API integration for movie data
- Mixed precision inference (FP16) for speed
- Safetensors model format for 2x faster loading
- Performance: ~300ms latency per request

Frontend (React + Vite on Vercel):
- Responsive design optimized for mobile and desktop
- Dark/Light theme toggle
- Real-time emotion detection with visual feedback
- Movie carousels organized by genre
- Example text suggestions for users
- API health monitoring

Emotion-to-Genre Mapping:
- Anger → Action, Crime, Thriller, Revenge-Drama
- Fear → Horror, Thriller, Mystery, Supernatural
- Joy → Comedy, Adventure, Family, Animation, Musical
- Love → Romance, Rom-Com, Emotional Drama, Fantasy
- Neutral → Documentary, Drama, Biography, Slice-of-Life
- Sadness → Drama, Romance, Indie, Healing-Stories
- Surprise → Mystery, Sci-Fi, Fantasy, Twist-Thriller

**Evaluation & Testing:**

I created 12 comprehensive visualizations for model evaluation:
- Confusion matrices (raw and normalized)
- Per-class metrics (F1, Precision, Recall)
- ROC curves for each emotion
- Precision-Recall curves
- Error analysis showing misclassification patterns
- Train vs. test comparison

Most common misclassifications:
- Fear → Surprise (semantic similarity in unexpected events)
- Joy → Love (positive emotion overlap)
- Surprise → Neutral (ambiguous neutral reactions)

These patterns reveal linguistic challenges where emotions share semantic features.

**Deployment:**
- Frontend: https://moodflix-ai-nu.vercel.app
- API: https://mahmdshafee-emotion-detection-api.hf.space

**What Made This Project Special:**

This wasn't just a model. It was a complete system. I had to think about:
- Data quality and balance
- Model architecture and training
- Inference optimization
- API design
- Frontend user experience
- Deployment infrastructure
- Monitoring and reliability

I learned that production ML is different from academic ML. You can't just train a model and call it done. You have to think about latency, memory usage, error handling, and user experience.

**Key Insight:** Building a complete system teaches you things that building just a model never will. You understand trade-offs, constraints, and the importance of every layer.

---

## Individual Projects

Throughout the fellowship, I completed several solo projects that allowed me to explore specific areas deeply:

1. **Week 1-4:** Weekly assignments and tasks
2. **Week 3:** Feedback Dashboard & Report
3. **Week 6:** Book Recommendation System
4. **Week 8:** Deep Learning Assignment
5. **Week 9:** Iris Clustering Analysis
6. **Week 10:** Cluster Analysis Comparison
7. **Week 9-12:** MoodFlix AI (Final Project)

These projects taught me self-directed learning, problem-solving, and how to take feedback and iterate.

---

## Peer Collaboration

### Food Delivery Delay Prediction (Weeks 5-6)

This was my first experience leading a technical team. I learned that leadership isn't about knowing everything—it's about:
- Setting clear goals and timelines
- Dividing work based on strengths
- Communicating progress and blockers
- Making decisions when faced with uncertainty
- Adapting when circumstances change (like when a team member had to leave)

The project succeeded because we focused on delivering value rather than perfection. We built something that worked, was understandable, and could be deployed.

---

## Hackathon Experience

### ExoVision AI (Week 7)

The hackathon was intense and exhilarating. In a compressed timeframe, we built a complete ML system with a web interface. I learned:
- How to work under pressure
- The importance of clear communication in a team
- How to make quick decisions with incomplete information
- That sometimes "good enough" is better than "perfect"
- The value of diverse skills on a team

The hackathon taught me that I could handle complexity and ambiguity. It built confidence that I could tackle real-world problems.

---

## Final Capstone Project

MoodFlix AI represents the culmination of 12 weeks of learning. It demonstrates:
- Deep learning expertise (DeBERTa fine-tuning)
- Full-stack development (FastAPI + React)
- Production deployment (HuggingFace Spaces + Vercel)
- Comprehensive evaluation (12 visualizations, detailed metrics)
- Real-world thinking (latency, memory, user experience)

This project is more than code. It's evidence that I can take an idea from concept to production, handling every layer of the stack.

---

## Key Technologies

### Programming Languages
- Python 3.8+
- JavaScript/React

### Data Science & ML
- NumPy, Pandas, Scikit-learn
- XGBoost, LightGBM
- Matplotlib, Seaborn
- Jupyter Notebooks

### Deep Learning
- PyTorch 2.2.0
- Transformers 4.37.2 (HuggingFace)
- DeBERTa v3 Base
- Safetensors

### Web Frameworks
- Flask (Python)
- FastAPI (Python)
- React 19.2 (JavaScript)
- Vite 7.2 (Build tool)
- TailwindCSS 4.1 (Styling)

### Deployment & Infrastructure
- HuggingFace Spaces (Docker)
- Vercel (Serverless)
- Git LFS (Large file storage)

### Data Visualization & BI
- Matplotlib, Seaborn
- Power BI
- Monday.com

### APIs & External Services
- TMDB API (Movie Database)
- NASA Exoplanet Archive
- HuggingFace Model Hub

---

## Datasets & Models

### Datasets Used
1. Iris Dataset - Classification, clustering
2. Breast Cancer Dataset - Classification
3. California Housing Dataset - Regression
4. Boston Housing Dataset - Regression
5. Titanic Dataset - Classification, model evaluation
6. Student Performance Dataset - Regression
7. Diabetes Dataset - Classification
8. Car Evaluation Dataset - Classification
9. Mall Customers Dataset - Clustering
10. Food Delivery Dataset - Custom, ~1000 records
11. NASA K2 Planets Catalog - Exoplanet classification
12. SuperEmotion Dataset - NLP, emotion detection (190,716 samples)

### Models Developed

**Regression:**
- Linear Regression
- XGBoost Regressor (Food Delivery: R² = 0.81)

**Classification:**
- Logistic Regression
- Decision Trees
- Random Forests
- SVM
- XGBoost Classifier (Food Delivery: 94.5% accuracy, ExoVision: 99% accuracy)
- LightGBM
- DeBERTa v3 (MoodFlix: 90.54% accuracy)

**Unsupervised Learning:**
- K-Means Clustering
- Hierarchical Clustering

**Recommendation Systems:**
- Collaborative Filtering
- Content-based Filtering

---

## Reflections & Growth

### Technical Growth

I started Week 1 knowing Python basics but uncertain about data science. By Week 12, I could:
- Build end-to-end ML pipelines
- Fine-tune state-of-the-art transformer models
- Deploy systems to production
- Evaluate models comprehensively
- Optimize for real-world constraints

The progression was deliberate. Each week built on the previous. Foundations first, then supervised learning, then unsupervised learning, then deep learning. By the time I reached the capstone, I had the tools to build something sophisticated.

### Professional Growth

Beyond technical skills, I learned:
- How to communicate complex ideas simply
- How to lead a team
- How to accept feedback and iterate
- How to manage time and prioritize
- How to think about trade-offs and constraints
- That data science is as much about business understanding as technical skill

### Personal Growth

The fellowship was challenging. There were moments of frustration, confusion, and self-doubt. But there were also moments of breakthrough, when something clicked and suddenly made sense. I learned that growth happens at the edge of your comfort zone, and that consistency matters more than brilliance.

I also learned the value of community. The other fellows were supportive, collaborative, and inspiring. We pushed each other to be better.

### What I'd Do Differently

If I could do it again:
- I'd ask more questions earlier
- I'd spend more time understanding why things work, not just how
- I'd document my learning more systematically
- I'd reach out for help sooner when stuck

But overall, I'm proud of the journey. It was real, it was challenging, and it was transformative.

---

## The Bigger Picture

This fellowship wasn't just about learning data science. It was about learning how to learn, how to solve problems, and how to grow. The specific technologies and datasets will change, but the thinking will remain.

I learned that data science is fundamentally about curiosity. It's about asking questions, exploring data, finding patterns, and translating those patterns into action. It's about being rigorous with your analysis while remaining humble about what you don't know.

I also learned that the best data scientists are those who can bridge the gap between technical complexity and business simplicity. Anyone can build a model. The skill is in building a model that matters, that gets used, and that creates value.

---

## Looking Forward

This fellowship was the beginning, not the end. I'm excited about:
- Diving deeper into NLP and transformers
- Exploring reinforcement learning
- Building more sophisticated recommendation systems
- Working on real-world problems with real impact
- Continuing to learn and grow

The fellowship gave me the foundation and the confidence to pursue these interests. More importantly, it taught me how to learn. That's the real skill.

---

## Acknowledgments

I'm grateful to:
- The Buildables team for creating an excellent program
- My team leads and mentors for guidance and feedback
- My fellow fellows for collaboration and support
- My teammates on the peer and hackathon projects for pushing me to be better

This journey was collaborative. I didn't do it alone, and I'm better for it.

---

## Repository Structure

```
.
├── Week01/                          # Foundations (Python, OOP, DSA, Data Science)
│   ├── Task01/                      # Python fundamentals (9 programs)
│   ├── Task02/                      # OOP & DSA
│   ├── Task03/                      # Advanced DSA
│   └── Task04/                      # Data science intro (12 scripts)
│
├── Week02/                          # Statistics & Data Preprocessing
│   ├── Task01/                      # Descriptive & Inferential Statistics
│   ├── Task02/                      # Data manipulation (20 tasks)
│   └── Task03/                      # Feature scaling & imputation
│
├── Week03 - Buildables Fellows Feedback Dashboard And Report/
│   ├── data/                        # Cleaned feedback data
│   ├── dashboard/                   # Monday.com dashboard (PDF)
│   └── reports/                     # 6-page analytical report
│
├── Week04/                          # Supervised Learning
│   ├── Task01/                      # Linear regression
│   ├── Task02/                      # Classification
│   ├── Task03/                      # Multi-algorithm comparison
│   └── Task04/                      # Model evaluation
│
├── Week05-06 (Peer Project) - Food Delivery Delay Prediction/
│   ├── app/                         # Flask web application
│   ├── data/                        # Raw, processed, and engineered data
│   ├── models/                      # Trained XGBoost models
│   ├── notebooks/                   # 5 Jupyter notebooks
│   └── dashboard/                   # Power BI dashboard
│
├── Week06 - Daily Tasks/            # Book Recommendation System
│   └── BookRecommendationSystem.ipynb
│
├── Week07 - ExoVision AI/           # NASA Space Apps Hackathon 2025
│   ├── app/                         # Flask web application
│   ├── Dataset/                     # NASA K2 catalog
│   ├── Models/                      # Trained models (Decision Tree, XGBoost, LightGBM)
│   ├── Notebook/                    # Model training & evaluation
│   └── presentation/                # Hackathon presentation
│
├── Week08 - Deep Learning/          # Deep learning theory & assignment
│   ├── deep_learning.ipynb
│   └── assignment.pdf
│
├── Week09 (Daily Tasks) - Iris Clustering Analysis/
│   └── kmeans_analysis.ipynb
│
├── Week10 (Daily Tasks) - Cluster Analysis Comparison/
│   └── cluster-analysis.ipynb
│
├── Week09-12 (FINAL PROJECT) - MoodFlix AI Movie Recommendation System/
│   ├── app/
│   │   ├── backend/                 # FastAPI application
│   │   └── frontend/                # React + Vite application
│   ├── data/                        # SuperEmotion dataset (190,716 samples)
│   ├── models/                      # DeBERTa v3 model & weights
│   ├── notebooks/                   # Data cleaning, training, testing
│   ├── DeBERTa Test Results/        # 12 evaluation visualizations
│   └── Dockerfile                   # HuggingFace Spaces deployment
│
├── BLOGS.md                         # Published Medium articles
└── README.md                        # This file
```

---

## Contact & Connect

I'm always interested in discussing data science, machine learning, and the journey of learning. Feel free to reach out:

- **LinkedIn:** https://www.linkedin.com/in/muhammad-ahmad-shafee
- **Medium:** https://medium.com/@mahmdshafee
- **GitHub:** https://github.com/c0llectorr

---

## Final Thoughts

This fellowship was one of the most challenging and rewarding experiences of my life. It pushed me to grow, taught me to think differently, and showed me what's possible when you commit to learning.

If you're considering a data science fellowship or career, my advice is simple: start. Don't wait until you feel ready. You'll never feel completely ready. Start with the fundamentals, build projects, learn from failures, and keep iterating. The journey is the destination.

Thank you to everyone who was part of this journey. Here's to the next chapter.

---

**Last Updated:** January 2026  
**Fellowship Duration:** 12 weeks (Remote)  
**Program:** Buildables Data Science Fellowship
