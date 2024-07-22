import streamlit as st
import pandas as pd
from transformers import BertTokenizer, BertModel, RobertaTokenizer, RobertaModel
from sklearn.decomposition import PCA
import plotly.graph_objs as go
import numpy as np
from database_utils import init_db, save_embeddings_to_db, get_all_embeddings, clear_all_entries, fetch_data_as_csv

@st.cache_resource
def load_model(model_name):
    if model_name == "BERT":
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
    elif model_name == "RoBERTa":
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaModel.from_pretrained('roberta-base')
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    return tokenizer, model

def get_embeddings(phrases, tokenizer, model):
    embeddings = []
    for phrase in phrases:
        inputs = tokenizer(phrase, return_tensors='pt', padding=True, truncation=True)
        outputs = model(**inputs)
        mean_embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy()
        embeddings.append(mean_embedding[0])
    return np.array(embeddings)

def plot_interactive_embeddings(embeddings, phrases):
    if len(phrases) >= 2:
        pca = PCA(n_components=min(3, len(phrases)))
        reduced_embeddings = pca.fit_transform(embeddings)
        
        if len(phrases) == 2:
            fig = go.Figure(data=[
                go.Scatter(x=[emb[0]], y=[emb[1]], mode='markers+text', text=[phrase], name=phrase)
                for emb, phrase in zip(reduced_embeddings, phrases)
            ])
            fig.update_layout(title='2D Scatter Plot of Embeddings', xaxis_title='PCA Component 1', yaxis_title='PCA Component 2')
        else:
            fig = go.Figure(data=[
                go.Scatter3d(x=[emb[0]], y=[emb[1]], z=[emb[2]], mode='markers+text', text=[phrase], name=phrase)
                for emb, phrase in zip(reduced_embeddings, phrases)
            ])
            fig.update_layout(title='3D Scatter Plot of Embeddings',
                              scene=dict(xaxis_title='PCA Component 1', yaxis_title='PCA Component 2', zaxis_title='PCA Component 3'))
        
        fig.update_layout(autosize=False, width=800, height=600)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Please add at least one more phrase to visualize.")

def main():
    st.set_page_config(layout="wide")
    st.title("Language Model Embeddings Visualization")

    st.markdown("""
    This application visualizes embeddings of words and phrases from BERT or RoBERTa language models. 
    Explore how different words and phrases relate to each other in the embedding space!
    """)

    # Load model at the beginning
    model_choice = "BERT"  # Default model
    tokenizer, model = load_model(model_choice)

    
    # Sidebar
    with st.sidebar:
        st.header("Controls")
        model_choice = st.selectbox("Choose a model:", ["BERT", "RoBERTa"])

        if model_choice != st.session_state.get('last_model_choice'):
            tokenizer, model = load_model(model_choice)
            st.session_state.last_model_choice = model_choice
        
        new_phrase = st.text_input("Enter a new word or phrase:", "")
        if st.button("Add Phrase"):
            if new_phrase and new_phrase not in st.session_state.phrases:
                embedding = get_embeddings([new_phrase], tokenizer, model)[0]
                save_embeddings_to_db(new_phrase, embedding)
                st.session_state.phrases.append(new_phrase)
                st.experimental_rerun()
        
        uploaded_file = st.file_uploader("Upload CSV file", type="csv")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            phrase_column = next((col for col in ['phrase', 'Phrase'] if col in df.columns), None)
            if phrase_column:
                new_phrases = df[phrase_column].dropna().unique().tolist()
                for phrase in new_phrases:
                    if phrase and phrase not in st.session_state.phrases:
                        embedding = get_embeddings([phrase], tokenizer, model)[0]
                        save_embeddings_to_db(phrase, embedding)
                        st.session_state.phrases.append(phrase)
                st.success(f"Successfully imported {len(new_phrases)} new phrases.")
                st.experimental_rerun()
            else:
                st.error("The CSV file must contain a 'phrase' or 'Phrase' column.")
        
        if st.button("Clear All Entries"):
            clear_all_entries()
            st.session_state.phrases = [default_phrase]
            embedding = get_embeddings([default_phrase], tokenizer, model)[0]
            save_embeddings_to_db(default_phrase, embedding)
            st.experimental_rerun()
        
        if st.button("Download Database as CSV"):
            csv = fetch_data_as_csv()
            st.download_button(label="Download CSV", data=csv, file_name='embeddings.csv', mime='text/csv')

    # Main area
    tokenizer, model = load_model(model_choice)

    default_phrase = "example"
    if "phrases" not in st.session_state:
        st.session_state.phrases = [default_phrase]
        init_db()
        embedding = get_embeddings([default_phrase], tokenizer, model)[0]
        save_embeddings_to_db(default_phrase, embedding)

    st.subheader(f"Current phrases ({model_choice}):")
    st.write(", ".join(st.session_state.phrases))

    embeddings, phrases = get_all_embeddings()
    if len(embeddings) > 0:
        embeddings = np.array(embeddings)
        plot_interactive_embeddings(embeddings, phrases)
    else:
        st.info("Add phrases using the sidebar to visualize their embeddings.")

if __name__ == "__main__":
    main()