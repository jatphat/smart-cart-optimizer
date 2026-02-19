

# ===========================================
# IMPORTS AND INITIAL SETUP
# ===========================================
import pandas as pd
import numpy as np
import streamlit as st
from openai import OpenAI
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
import uuid
import os
from dotenv import load_dotenv
from html import escape

# Load .env variables (for local use)
load_dotenv()

# Configure Streamlit page
st.set_page_config(
    page_title="Smart Cart Optimizer",
    page_icon="🛒",
    layout="wide"
)
st.markdown("""
    <style>
        .stChatMessage {
            padding: 12px;
            border-radius: 8px;
            margin-bottom: 15px;
        }
        .stChatMessage p {
            margin-bottom: 4px;
        }
        [data-testid="stChatInput"] {
            background-color: #f8f9fa;
            border-radius: 8px;
            padding: 12px;
        }
    </style>
""", unsafe_allow_html=True)

# ✅ Secure API key fallback logic
api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# ✅ Example usage (test connection)
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hi there!"}]
)
st.write(response.choices[0].message.content)


# ===========================================
# DATA LOADING AND PREPROCESSING
# ===========================================
@st.cache_data
def load_data():
    """Load and preprocess the dataset"""
    try:
        # Load the data
        # Get the directory where the script is located
        base_path = os.path.dirname(__file__)
        file_path = os.path.join(base_path, "Enhanced_Walmart_Data_Fixed_NoOutliers.csv")

        # Load the data
        df = pd.read_csv(file_path, header=1)
        #df = pd.read_csv("Enhanced_Walmart_Data_Fixed_NoOutliers.csv", header=1)
        
        # Basic preprocessing
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['Sale Price'] = pd.to_numeric(df['Sale Price'], errors='coerce').fillna(0)
        
        # Clean up names
        df['name'] = df['name'].fillna('Guest User')
        df['name'] = df['name'].apply(lambda x: x if len(str(x)) > 1 else 'Guest User')
        
        # Create combined text for product similarity
        df['combined_text'] = (
            df['Product Name'].fillna('') + ' ' +
            df['category'].fillna('') + ' ' +
            df['brand'].fillna('') + ' ' +
            df['Label'].fillna('')
        )
        
        return df
    
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Load the data
df = load_data()

if df is None:
    st.error("Failed to load data. Please check your data file.")
    st.stop()

# ===========================================
# CART MANAGEMENT CLASS
# ===========================================
class ShoppingCart:
    def __init__(self):
        if 'cart' not in st.session_state:
            st.session_state.cart = []
    
    def add_item(self, product_info):
        """Add an item to the cart"""
        if product_info and not any(
            item['Product Name'] == product_info['Product Name'] 
            for item in st.session_state.cart
        ):
            st.session_state.cart.append(product_info)
    
    def remove_item(self, index):
        """Remove an item from the cart"""
        if 0 <= index < len(st.session_state.cart):
            st.session_state.cart.pop(index)
            st.rerun()
    
    def get_total(self):
        """Calculate cart total"""
        return sum(item['Sale Price'] for item in st.session_state.cart)
    
    def clear(self):
        """Clear the cart"""
        st.session_state.cart = []
        st.rerun()

# Initialize shopping cart
cart = ShoppingCart()

# ===========================================
# RECOMMENDATION SYSTEM
# ===========================================
@st.cache_resource
def initialize_recommendation_model(df):
    """Initialize the recommendation model"""
    try:
        # Create TF-IDF vectorizer for product similarity
        tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
        tfidf_matrix = tfidf.fit_transform(df['combined_text'])
        
        # Create KNN model for finding similar products
        knn_model = NearestNeighbors(n_neighbors=5, metric='cosine')
        knn_model.fit(tfidf_matrix)
        
        return tfidf, knn_model
    except Exception as e:
        st.error(f"Error initializing recommendation model: {str(e)}")
        return None, None

def get_smart_recommendations(user_id, cart_items, cart_total, threshold, df, n_recommendations=5):
    try:
        gap = max(0, threshold - cart_total)
        
        # Get current cart categories
        cart_categories = [item['category'] for item in cart_items]
        
        # Filter items within price range and NOT in cart categories
        potential_recs = df[
            (df['Sale Price'] >= (gap - 5)) &
            (df['Sale Price'] <= (gap + 5)) &
            (~df['category'].isin(cart_categories))  # Exclude cart categories
        ].copy()
        
        # If not enough recommendations, expand price range
        if len(potential_recs) < n_recommendations:
            potential_recs = df[
                (df['Sale Price'] <= gap + 10) &
                (~df['category'].isin(cart_categories))
            ].copy()
        
        # Score recommendations
        potential_recs['score'] = 0
        
        # Scoring factors
        potential_recs['price_score'] = 1 - abs(potential_recs['Sale Price'] - gap) / gap
        potential_recs['discount_score'] = potential_recs['Label'].str.lower().isin(
            ['rollback', 'clearance', 'discount']
        ).astype(int)
        
        # Calculate final score
        potential_recs['score'] = (
            potential_recs['price_score'] * 0.6 +
            potential_recs['discount_score'] * 0.4
        )
        
        # Get diverse recommendations (one per category if possible)
        final_recommendations = (
            potential_recs
            .sort_values('score', ascending=False)
            .groupby('category')
            .first()
            .reset_index()
            .head(n_recommendations)
        )
        
        return final_recommendations
        
    except Exception as e:
        st.error(f"Error generating recommendations: {str(e)}")
        return pd.DataFrame()


# def format_item(item):
#     return f"""
#     <div style='margin-bottom:15px; padding:12px; border:1px solid #e0e0e0; border-radius:8px; background-color:#f8f9fa'>
#         <strong style='font-size:1.1em'>{item['Product Name']}</strong><br>
#         <span style='color:green; font-weight:bold'>${item['Sale Price']:.2f}</span><br>
#         <span style='color:#666; font-size:0.9em'>{item['category']} • {item['Label']}</span>
#     </div>
  
def format_item(item):
    name = escape(str(item['Product Name']))
    category = escape(str(item['category']))
    label = escape(str(item['Label']))
    
    return f"""
    <div style='margin-bottom:15px; padding:12px; border:1px solid #e0e0e0; border-radius:8px; background-color:#f8f9fa'>
        <strong style='font-size:1.1em'>{name}</strong><br>
        <span style='color:green; font-weight:bold'>${item['Sale Price']:.2f}</span><br>
        <span style='color:#666; font-size:0.9em'>{category} • {label}</span>
    </div>
    """

    
# ===========================================
# AI SHOPPING ASSISTANT
# ===========================================
@st.cache_data(ttl=3600)
def get_shopping_assistant_response(query, cart_items, df):
    """
    AI assistant that responds using shop inventory first,
    and only falls back to general advice if needed.
    """

    from collections import defaultdict
    import re

    try:
        query_lower = query.lower()
        common_words = {'any', 'show', 'me', 'in', 'on', 'at', 'the', 'for', 'items', 'products', 'a', 'i'}
        search_terms = [word for word in query_lower.split() if word not in common_words and len(word) > 2]

        
        # 🎁 1. Gift-related query
        if any(term in query_lower for term in ['gift', 'father', "dad", "father's day"]):
            gift_categories = ["Electronics", "Accessories", "Sports", "Health", "Home"]
            gift_items = df[
                (df['category'].isin(gift_categories)) &
                (df['Label'].str.lower().isin(['discount', 'rollback', 'clearance'])) &
                (df['Sale Price'] > 1)
            ].drop_duplicates(subset=["Product Name"]).sort_values("Sale Price").head(5)

            if not gift_items.empty:
                html_blocks = "".join([format_item(row) for _, row in gift_items.iterrows()])
                response = f"""
                <div style='padding:10px'>
                    <h3 style='color:#2e86c1; margin-bottom:20px'>🎁 Here are some gift ideas from our store:</h3>
                    {html_blocks}
                </div>
                """
                return response

       

        # 💰 2. Cheapest items query
        if any(term in query_lower for term in ['cheapest', 'lowest', 'on sale']):
            cheapest_items = df[df['Sale Price'] > 0].drop_duplicates(subset=["Product Name"]).sort_values("Sale Price").head(5)
                         
            if not cheapest_items.empty:
                html_blocks = "".join([format_item(row) for _, row in cheapest_items.iterrows()])
                response = f"""
                <div style='padding:10px'>
                    <h3 style='color:#27ae60; margin-bottom:20px'>💰 Here are the cheapest items available right now:</h3>
                    {html_blocks}
                </div>
                """
                return response

        # 🏠 3. Category-based query
        for cat in df['category'].dropna().unique():
            if cat.lower() in query_lower:
                matched_items = df[
                    (df['category'].str.lower() == cat.lower()) &
                    (df['Sale Price'] > 0)
                ].drop_duplicates(subset=["Product Name"]).sort_values("Sale Price").head(5)

                if not matched_items.empty:
                    html_blocks = "".join([format_item(row) for _, row in matched_items.iterrows()])
                    response = f"""
                    <div style='padding:10px'>
                        <h3 style='color:#9b59b6; margin-bottom:20px'>🛍️ Here are some {cat} items from our store:</h3>
                        {html_blocks}
                    </div>
                    """
                    return response

        


        # 💸 4. Price-based query e.g. "under $10"
        match = re.search(r'under\s?\$?(\d+)', query_lower)
        if match:
            max_price = float(match.group(1))
            filtered_items = df[
                (df['Sale Price'] > 0) & (df['Sale Price'] <= max_price)
            ].drop_duplicates(subset=["Product Name"]).sort_values("Sale Price").head(5)

            if not filtered_items.empty:
                # response = f"💸 **Items under ${max_price:.2f}:**\n\n"
                # response += "\n".join([format_item(row) for _, row in filtered_items.iterrows()])
                # return response
                response = f"<h4>💸 Items under <span style='color:green;'>${max_price:.2f}</span>:</h4>"
                response += "".join([format_item(row) for _, row in filtered_items.iterrows()])
                return response  
            # return f"⚠️ Sorry, no items found under ${max_price:.2f}."
            return f"<p>⚠️ Sorry, no items found under <strong>${max_price:.2f}</strong>.</p>"



        # 🤖 5. Fallback to OpenAI if nothing relevant found
        system_prompt = f"""
You are a helpful and honest shopping assistant for a real online store.

✅ First, always try to answer based on the store's inventory.

❌ Do NOT hallucinate products or categories.

🛒 Available product categories: {', '.join(sorted(df['category'].dropna().unique()))}
🛍️ Cart subtotal: ${sum(item['Sale Price'] for item in cart_items):.2f}
🚚 Shipping threshold: $35.00

Only provide general advice if you cannot find anything relevant in the inventory.
Be concise, friendly, and always helpful.
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=200,
            temperature=0.7
        )

        return "🤖 " + response.choices[0].message.content.strip()

    except Exception as e:
        st.error(f"Error: {str(e)}")
        return "⚠️ Sorry, I couldn't process your question right now. Please try again."


# ===========================================
# STREAMLIT INTERFACE
# ===========================================
def main():
    st.title("🛒 Smart Cart Optimizer")
    
    # ===========================================
    # USER SELECTION
    # ===========================================
    col1, col2 = st.columns([3, 1])
    with col1:
        # Create user selection with better name handling
        unique_users = df[['CustID', 'name']].drop_duplicates().sort_values('CustID')
        user_options = [
            f"{row['CustID']} - {row['name'] if pd.notna(row['name']) and len(str(row['name'])) > 1 else 'Guest User'}"
            for _, row in unique_users.iterrows()
        ]
        selected_user = st.selectbox("Select Your Customer ID", user_options)
        selected_user_id = int(selected_user.split(" - ")[0])
        user_name = selected_user.split(" - ")[1]
        
    with col2:
        if st.button("Clear Cart 🗑️"):
            cart.clear()
    
    st.write(f"Welcome back, **{user_name}**! 👋")
    
    # ===========================================
    # PRODUCT SELECTION
    # ===========================================
    st.subheader("Add Products to Cart")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Category selection
        categories = sorted(df['category'].unique())
        selected_category = st.selectbox("Select Category", categories)
    
    with col2:
        # Product multiselect within category
        filtered_products = df[df['category'] == selected_category]['Product Name'].unique()
        selected_products = st.multiselect("Select Products", filtered_products)
        
        # Add button for all selected products
        if selected_products:
            col1, col2 = st.columns([3, 1])
            with col1:
                # Display selected products info
                for product in selected_products:
                    product_info = df[df['Product Name'] == product].iloc[0]
                    st.write(f"• {product}: ${product_info['Sale Price']:.2f}")
            with col2:
                if st.button("Add Selected to Cart 🛒"):
                    for product in selected_products:
                        product_info = df[df['Product Name'] == product].iloc[0].to_dict()
                        cart.add_item(product_info)
                    st.rerun()
    
    # ===========================================
    # CART DISPLAY
    # ===========================================
    st.markdown("---")
    st.subheader("Your Shopping Cart")
    
    if st.session_state.cart:
        # Style for cart table
        st.markdown("""
            <style>
            .cart-header {
                font-weight: bold;
                border-bottom: 2px solid #808080;
                padding: 10px 0;
                margin-bottom: 10px;
            }
            .cart-row {
                padding: 8px 0;
                border-bottom: 1px solid #eee;
            }
            </style>
        """, unsafe_allow_html=True)
        
        # Cart header
        cols = st.columns([3, 2, 2, 1])
        cols[0].markdown('<p class="cart-header">Product</p>', unsafe_allow_html=True)
        cols[1].markdown('<p class="cart-header">Category</p>', unsafe_allow_html=True)
        cols[2].markdown('<p class="cart-header">Price</p>', unsafe_allow_html=True)
        cols[3].markdown('<p class="cart-header">Remove</p>', unsafe_allow_html=True)
        
        # Display cart items
        cart_total = 0
        for idx, item in enumerate(st.session_state.cart):
            cart_total += item['Sale Price']
            
            cols = st.columns([3, 2, 2, 1])
            with cols[0]:
                st.markdown(f"<p class='cart-row'>{item['Product Name']}</p>", unsafe_allow_html=True)
            with cols[1]:
                st.markdown(f"<p class='cart-row'>{item['category']}</p>", unsafe_allow_html=True)
            with cols[2]:
                st.markdown(f"<p class='cart-row'>${item['Sale Price']:.2f}</p>", unsafe_allow_html=True)
            with cols[3]:
                if st.button("❌", key=f"remove_{idx}_{hash(item['Product Name'])}"):
                    cart.remove_item(idx)
        
        # Cart summary
        st.markdown("---")
        st.markdown(f"**Cart Total:** ${cart_total:.2f}")
        
        # Shipping threshold info
        threshold = 35
        if cart_total >= threshold:
            st.success("🎉 Congratulations! Your order qualifies for free shipping!")
        else:
            remaining = threshold - cart_total
            st.warning(f"Add ${remaining:.2f} more to your cart to qualify for free shipping!")
            
            # Show smart recommendations
            st.subheader("Recommended Items to Reach Free Shipping")
            recommendations = get_smart_recommendations(
                user_id=selected_user_id,
                cart_items=st.session_state.cart,
                cart_total=cart_total,
                threshold=threshold,
                df=df
            )
            
            
            if not recommendations.empty:
                for index, rec in recommendations.iterrows():
                    cols = st.columns([3, 2, 2, 1])
                    with cols[0]:
                        st.write(rec['Product Name'])
                    with cols[1]:
                        st.write(rec['category'])
                    with cols[2]:
                        st.write(f"${rec['Sale Price']:.2f} ({rec['Label']})")
                    with cols[3]:
                        unique_key = f"rec_{index}_{hash(rec['Product Name'] + str(rec['Sale Price']))}"
                        if st.button("Add 📦", key=unique_key):
                            product_info = {
                                'Product Name': rec['Product Name'],
                                'category': rec['category'],
                                'Sale Price': rec['Sale Price'],
                                'Label': rec['Label'],
                                'ProductID': df[df['Product Name'] == rec['Product Name']].iloc[0]['ProductID']
                            }
                            cart.add_item(product_info)
                            st.rerun()  # Force refresh after adding item
                
    else:
        st.info("Your cart is empty. Start shopping by selecting products above!")
    
    # ===========================================
    # AI SHOPPING ASSISTANT
    # ===========================================
        # ===========================================
    # AI SHOPPING ASSISTANT
    # ===========================================
    st.markdown("---")
    st.subheader("Shopping Assistant")
    
    # Show capabilities
    with st.expander("What can I help you with?"):
        st.write("• Finding specific products and their prices")
        st.write("• Checking for deals and discounts")
        st.write("• Suggesting items to reach free shipping")
        st.write("• Answering general shopping questions")

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history
    
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            if isinstance(message["content"], str) and ("<div" in message["content"] or "<h" in message["content"]):
                st.markdown(message["content"], unsafe_allow_html=True)
            else:
                st.markdown(message["content"])

    # Query input
    
    #query = st.chat_input("Type your question here...", key="query_input")
    query = st.chat_input("Type your question here...")

   
    if query:
    # Display user message first
        with st.chat_message("user"):
            st.markdown(query)
        
        # Then get and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = get_shopping_assistant_response(query, st.session_state.cart, df)
                st.markdown(response, unsafe_allow_html=True)
        
        # Add to chat history
        st.session_state.chat_history.append({"role": "user", "content": query})
        st.session_state.chat_history.append({"role": "assistant", "content": response})

    # Clear chat button
    if st.session_state.chat_history:
        if st.button("Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

if __name__ == "__main__":

    main()
