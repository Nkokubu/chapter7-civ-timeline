import requests, streamlit as st, plotly.express as px

st.set_page_config(page_title="Civilization Timeline", layout="wide")
st.title("Civilization Timeline (alpha)")

with st.sidebar:
    st.header("Filters")
    year_range = st.slider("Year range (BCE negative)", -3000, 2000, (-500, 500), step=50)
    regions = st.multiselect("Regions", ["Europe","East Asia","South Asia","Middle East","Africa","Americas","Oceania"])

base = "http://127.0.0.1:8000"
resp = requests.get(f"{base}/civs", params={"start": year_range[0], "end": year_range[1], "region": regions})
items = resp.json()["items"]

st.subheader(f"Matching civilizations: {len(items)}")
if items:
    fig = px.timeline(
        x_start=[max(c["start_year"], year_range[0]) for c in items],
        x_end=[min(c["end_year"], year_range[1]) for c in items],
        y=[c["name"] for c in items]
    )
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No results in this range/region.")