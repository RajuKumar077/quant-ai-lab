import streamlit as st

st.set_page_config(
    page_title="Quant AI Lab",
    layout="wide"
)

st.title("📊 Quant AI Lab")
st.markdown("AI + Quant Finance Dashboard")

# Sidebar
st.sidebar.header("User Input")
ticker = st.sidebar.text_input("Enter Stock Ticker", "AAPL")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Stock Prediction",
    "📊 Portfolio Optimizer",
    "🎲 Monte Carlo",
    "⚠️ Value at Risk (VaR)"
])

with tab1:
    st.subheader("Stock Prediction Module")
    st.info("Model will be added here.")

with tab2:
    st.subheader("Portfolio Optimization Module")
    st.info("Optimizer will be added here.")

with tab3:
    st.subheader("Monte Carlo Simulation Module")
    st.info("Simulation logic will be added here.")

with tab4:
    st.subheader("Value at Risk (VaR) Module")
    st.info("Risk calculation will be added here.")
