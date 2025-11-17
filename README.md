## Background

In India, mental healthcare faces significant challenges with limited accessibility, widespread stigma, and critical resource gaps. The country has fewer than 1 psychiatrist per 100,000 people, creating immense barriers for those seeking timely support. Many individuals struggle to navigate complex healthcare systems while dealing with cultural taboos surrounding mental health discussions. 

This crisis is particularly acute among students facing academic pressure, LGBTQ+ communities experiencing discrimination, and rural populations with minimal access to services. Existing digital solutions often lack cultural context and fail to provide immediate, relevant resource matching. 

This AI-powered Mental Health Triage System addresses these gaps by offering instant, culturally-sensitive classification of mental health needs and connecting users with appropriate Indian mental health resources. The platform bridges critical accessibility divides while maintaining privacy and eliminating data storage concerns, making mental health support more approachable and actionable for diverse populations across the country.

----------------------------------------------------------------------------------------------------------

## Overview

Developed an intelligent triage platform that connects users with appropriate mental health support through:

* Resource Database: Added manually verified Indian mental health services with detailed metadata including specialties, languages, costs, and availability

* ML Classification: Trained a logistic regression model to distinguish crisis situations from therapy needs using TF-IDF vectorization and custom mental health lexicons

* User Interface: Built an interactive Streamlit web application featuring real-time analysis, smart filtering, and crisis detection with immediate emergency resource prioritization

* Deployment: Implemented end-to-end solution deployed via Streamlit Cloud with seamless ML model integration and responsive design for accessibility across devices

----------------------------------------------------------------------------------------------------------

## Link to test the web application

[Mental Health Triage System - website link](https://mental-health-india.streamlit.app)