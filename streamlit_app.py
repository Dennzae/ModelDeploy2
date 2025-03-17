import streamlit as st
import joblib

def main():
  st.title('Dermatology Machine Learning')
  st.info('This Using Machine learning')

  erythema = st.slider('Erythema',min_value=0,max_value=3,value=2)

if __name__ == "__main__":
  main()


