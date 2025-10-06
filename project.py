import pandas as pd
import streamlit as st
import plotly.graph_objects as go  
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------------------------------
# Load dataset
# -----------------------------------------------------
@st.cache_data
def load_data():
    dataset_path = r"AI_Resume_Screening.csv"
    df = pd.read_csv(dataset_path)
    df["Recruiter Decision"] = df["Recruiter Decision"].map({"Hire":1,"Reject":0})
    df["resume_text"] = (
        df["Skills"].fillna('') + " " +
        df["Education"].fillna('') + " " +
        df["Certifications"].fillna('') + " " +
        df["Job Role"].fillna('')
    )
    return df

df = load_data()

# -----------------------------------------------------
# Train ML model
# -----------------------------------------------------
@st.cache_resource
def train_model(df):
    text_features = "resume_text"
    num_features = ["Experience (Years)","Projects Count","Salary Expectation ($)","AI Score (0-100)"]

    train_df,test_df = train_test_split(df,test_size=0.3,random_state=42)
    preprocessor = ColumnTransformer([
        ("text",TfidfVectorizer(),text_features),
        ("num",StandardScaler(),num_features)
    ])
    model = Pipeline([
        ("features",preprocessor),
        ("clf",LogisticRegression(max_iter=300))
    ])
    model.fit(train_df[[text_features]+num_features],train_df["Recruiter Decision"])
    y_test = test_df["Recruiter Decision"]
    y_pred = model.predict(test_df[[text_features]+num_features])
    report = classification_report(y_test,y_pred,output_dict=True)
    return model,preprocessor,report

model, preprocessor, report = train_model(df)

# -----------------------------------------------------
# Streamlit UI
# -----------------------------------------------------
st.title("Resume Shortlisting System")
st.write("NLP & ML Based Ranking")

# Sidebar concise metrics
st.sidebar.subheader("ML Metrics")
st.sidebar.write(f"Hire Accuracy = {report['1']['f1-score']:.2f}")
st.sidebar.write(f"Reject Accuracy = {report['0']['f1-score']:.2f}")
st.sidebar.write(f"Overall Accuracy = {report['accuracy']:.2f}")

# Recruiter inputs
query = st.text_input("Enter job role / skills:")
alpha = st.slider("ML vs NLP Weight (0 = NLP, 1 = ML)",0.0,1.0,0.6)
top_n = st.slider("Number of candidates to show",1,20,5)

if st.button("Search Candidates"):
    # NLP similarity
    query_vec = preprocessor.named_transformers_["text"].transform([query])
    resume_vecs = preprocessor.named_transformers_["text"].transform(df["resume_text"])
    similarity_scores = cosine_similarity(query_vec,resume_vecs).flatten()

    # ML probability
    ml_probs_all = model.predict_proba(df[["resume_text","Experience (Years)","Projects Count",
                                           "Salary Expectation ($)","AI Score (0-100)"]])[:,1]

    # Hybrid score
    df["Hybrid_Score"] = alpha*ml_probs_all + (1-alpha)*similarity_scores
    ranked = df.sort_values("Hybrid_Score",ascending=False).reset_index(drop=True)
    ranked.index += 1

    # ---------------- Display top candidates with color ----------------
    st.subheader(f"Top {top_n} Candidates for '{query}'")
    for idx,row in ranked.head(top_n).iterrows():
        if idx == 1: 
            color = "#409628"; badge = "🥇"  # dark green
        elif idx == 2: 
            color = "#30B529"; badge = "🥈"  # Mild green
        elif idx == 3: 
            color = "#6BD747"; badge = "🥉"  # light green
        else:
            color = "#00AF83"; badge = f"{idx}"  # Aqua blue

        st.markdown(f"""
        <div style="background-color:{color}; padding:10px; border-radius:10px; margin-bottom:20px">
            <h3 style="margin:0">{badge} {row['Name']}</h3>
            <p style="margin:0"><strong>Score:</strong> {row['Hybrid_Score']:.2f}</p>
            <p style="margin:0"><strong>Skills:</strong> {row['Skills']}</p>
            <p style="margin:0"><strong>Experience:</strong> {row['Experience (Years)']} yrs | <strong>Projects:</strong> {row['Projects Count']} | <strong>Salary:</strong> ${row['Salary Expectation ($)']}</p>
        </div>
        """, unsafe_allow_html=True)

    # ---------------- Bar chart with matching colors ----------------
    colors = []
    for idx in ranked.head(top_n).index :
        if idx == 1:
            colors.append("#409628")
        elif idx == 2:
            colors.append("#30B529")
        elif idx == 3:
            colors.append("#6BD747")
        else:
            colors.append("#00AF83")

    fig = go.Figure(go.Bar(
        x=ranked.head(top_n)["Name"],
        y=ranked.head(top_n)["Hybrid_Score"],
        marker_color=colors
    ))
    fig.update_layout(
        title=f"📊 Top {top_n} Candidates Hybrid Score",
        xaxis_title="Candidate",
        yaxis_title="Hybrid Score",
        yaxis=dict(range=[0,1]),
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)
