
import streamlit as st
import requests


st.set_page_config(
    page_title="HealthTech Clinical NER Dashboard",
    page_icon="🏥",
    layout="wide"
)


API_URL = "http://127.0.0.1:8000/analyze"


st.title("🏥 Clinical Named Entity Recognition (NER)")
st.markdown(
    "Extract structured clinical concepts (**Problems**, **Treatments**, **Tests**) "
    "from unstructured physician narrative notes in real-time."
)

st.divider()


st.sidebar.header("System Status")
try:
    health_response = requests.get("http://127.0.0.1:8000/", timeout=3)
    if health_response.status_code == 200:
        st.sidebar.success("FastAPI Backend: ONLINE 🟢")
    else:
        st.sidebar.error("FastAPI Backend: UNHEALTHY 🔴")
except Exception:
    st.sidebar.error("FastAPI Backend: OFFLINE 🔴")
    st.sidebar.caption("Make sure Uvicorn is running on port 8000.")


default_text = (
    "Patient presents with severe migraine, persistent nausea, and chest pain. "
    "Administered 50mg Sumatriptan and ordered an ECG test. History of hypertension."
)

clinical_note = st.text_area(
    "Paste Unstructured Clinical Note Below:",
    value=default_text,
    height=150
)

col1, col2 = st.columns([1, 4])
with col1:
    analyze_button = st.button("Extract Entities 🚀", type="primary")


if analyze_button:
    if not clinical_note.strip():
        st.warning("Please enter clinical narrative text before analyzing.")
    else:
        with st.spinner("Processing clinical note with BERT model..."):
            try:
                
                payload = {"text": clinical_note}
                response = requests.post(API_URL, json=payload, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    entities = data.get("entities", [])
                    total_count = data.get("total_entities", 0)

                    st.success(f"Analysis Complete! Identified {total_count} clinical concepts.")
                    st.subheader("Extracted Entity Badges")

                    
                    CATEGORY_COLORS = {
                        "problem": "#FF4B4B",    
                        "treatment": "#1C83E1",  
                        "test": "#00C04D"       
                    }

                    
                    badge_html = ""
                    for item in entities:
                        entity_name = item["entity"]
                        category = item["category"].lower()
                        score = item["score"] * 100
                        bg_color = CATEGORY_COLORS.get(category, "#808080")

                        badge_html += f"""
                        <span style="
                            background-color: {bg_color};
                            color: white;
                            padding: 6px 12px;
                            margin: 4px;
                            border-radius: 16px;
                            display: inline-block;
                            font-weight: bold;
                            font-size: 14px;
                        ">
                            {entity_name.upper()} 
                            <span style="font-size: 11px; opacity: 0.85;">({category} | {score:.1f}%)</span>
                        </span>
                        """

                    st.markdown(badge_html, unsafe_allow_html=True)

                    st.divider()

                    
                    st.subheader("Detailed Category Breakdown")
                    st.dataframe(
                        entities,
                        column_config={
                            "entity": "Medical Entity",
                            "category": "Classification",
                            "score": st.column_config.NumberColumn(
                                "Confidence Score", format="%.4f"
                            )
                        },
                        use_container_width=True
                    )

                else:
                    st.error(f"API Error ({response.status_code}): {response.text}")

            except requests.exceptions.ConnectionError:
                st.error("Could not connect to FastAPI server. Ensure Uvicorn is running on http://127.0.0.1:8000.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")