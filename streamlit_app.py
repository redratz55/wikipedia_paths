import streamlit as st
import requests
from bs4 import BeautifulSoup
from queue import PriorityQueue
from sentence_transformers import SentenceTransformer, util

# --- Model Loading and Caching ---

# Use Streamlit's cache to load the model only once.
@st.cache_resource
def load_model():
    """Loads the sentence transformer model."""
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# --- A* Heuristic Function ---

def calculate_heuristic(page_title, target_title, target_embedding):
    """
    Calculates the heuristic cost between a page and the target using semantic similarity.
    The cost is lower for pages that are semantically closer to the target.
    """
    # We can improve performance by pre-calculating the target_embedding
    page_embedding = model.encode(page_title.replace("_", " "), convert_to_tensor=True)
    
    # Cosine similarity is between -1 and 1. Higher is better.
    cosine_sim = util.pytorch_cos_sim(page_embedding, target_embedding)[0][0].item()
    
    # A* needs a cost to minimize. We convert similarity to cost: 1 - similarity.
    # A similarity of 1 (identical) becomes a cost of 0.
    # A similarity of 0 (unrelated) becomes a cost of 1.
    cost = 1 - cosine_sim
    return cost

# --- A* Search Algorithm ---

def find_path_astar(start_page, end_page):
    """
    Finds the shortest path between two Wikipedia pages using A* search.
    """
    base_url = "https://en.wikipedia.org/wiki/"
    
    # Pre-calculate the embedding for the target page once
    target_embedding = model.encode(end_page.replace("_", " "), convert_to_tensor=True)

    # Priority queue stores tuples of (priority, path)
    # priority is f(n) = g(n) + h(n)
    queue = PriorityQueue()
    
    # g(n) is the length of the path
    g_cost = 0
    # h(n) is the heuristic cost
    h_cost = calculate_heuristic(start_page, end_page, target_embedding)
    # f(n) = g(n) + h(n)
    f_cost = g_cost + h_cost
    
    queue.put((f_cost, [start_page]))
    
    # Visited set stores tuples of (page, cost) to handle re-visiting with a lower cost
    visited = {start_page: 0}

    while not queue.empty():
        priority, path = queue.get()
        current_page = path[-1]
        
        # Current g(n) is the number of steps taken
        g_cost = len(path) - 1

        if current_page == end_page:
            return path # Goal found

        try:
            response = requests.get(f"{base_url}{current_page}")
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')

            for link in soup.select('p a[href^="/wiki/"]'):
                next_page = link.get('title')
                
                if next_page and not any(c in next_page for c in ':#'):
                    new_g_cost = g_cost + 1
                    
                    # If we've seen this page before but have now found a shorter path to it,
                    # we can explore from it again.
                    if next_page not in visited or new_g_cost < visited[next_page]:
                        visited[next_page] = new_g_cost
                        
                        h_cost = calculate_heuristic(next_page, end_page, target_embedding)
                        f_cost = new_g_cost + h_cost
                        
                        new_path = list(path)
                        new_path.append(next_page)
                        
                        queue.put((f_cost, new_path))
                        
        except requests.exceptions.RequestException as e:
            # This can happen for dead links or network issues
            print(f"Could not fetch {current_page}: {e}")
            continue

    return None # No path found

# --- Streamlit UI ---

st.title('Smarter Wikipedia Path Finder (A*) 🧠')
st.write("This app uses A* search and semantic similarity to find the shortest path between two Wikipedia articles.")

start_query = st.text_input("Start Page", "A.I. Winter")
end_query = st.text_input("End Page", "Neural network")

if st.button("Find Shortest Path"):
    if start_query and end_query:
        with st.spinner(f"Thinking... Searching for a path from '{start_query}' to '{end_query}'. This may take a moment."):
            # Format queries for Wikipedia URLs
            formatted_start = start_query.replace(" ", "_")
            formatted_end = end_query.replace(" ", "_")
            
            path = find_path_astar(formatted_start, formatted_end)

        st.success("Search complete!")

        if path:
            st.subheader("Shortest Path Found:")
            # Display path in a readable format
            display_path = " → ".join([p.replace("_", " ") for p in path])
            st.markdown(f"**{display_path}**")
            st.write(f"Path length: {len(path) - 1} clicks")
        else:
            st.warning("No path could be found.")
    else:
        st.error("Please enter both a start and an end page.")