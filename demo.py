import streamlit as st
import numpy as np

# Custom CSS for background and styling
st.markdown("""
    <style>
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }

        .stApp {
            animation: fadeIn 1s ease-in-out;
            background: linear-gradient(to right, #6a11cb, #2575fc);
            color: white;
        }

        .title {
            font-size: 32px;
            text-align: center;
            font-weight: bold;
            color: #ffffff;
            animation: fadeIn 1.5s;
        }

        .stButton>button {
            background-color: #ff9800;
            color: white;
            font-size: 18px;
            border-radius: 10px;
            width: 100%;
            padding: 10px;
            transition: all 0.3s ease-in-out;
        }

        .stButton>button:hover {
            background-color: #e68900;
            transform: scale(1.05);
        }

        .solution-box {
            padding: 15px;
            border-radius: 10px;
            background: rgba(255, 255, 255, 0.9);
            box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.2);
            margin-top: 20px;
            color: black;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="title">🔢 Linear Equation Solver</h1>', unsafe_allow_html=True)

# Function for Gaussian Elimination with Pivoting
def gaussian_elimination_pivoting(A, b):
    n = len(b)

    for i in range(n):
        max_row = i + np.argmax(np.abs(A[i:, i]))
        if i != max_row:
            A[[i, max_row]] = A[[max_row, i]]
            b[i], b[max_row] = b[max_row], b[i]
        
        diag_element = A[i][i]
        if abs(diag_element) < 1e-12:
            st.error("⚠️ Singular or nearly singular matrix detected. Cannot proceed.")
            return None

        A[i] = A[i] / diag_element
        b[i] = b[i] / diag_element

        for j in range(i + 1, n):
            factor = A[j][i]
            A[j] = A[j] - factor * A[i]
            b[j] = b[j] - factor * b[i]

    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = b[i] - np.dot(A[i][i + 1:], x[i + 1:])
    
    return x

# User Input
num_equations = st.number_input("📌 Enter the number of equations:", min_value=1, max_value=10, value=3, step=1)

equations = []
for i in range(int(num_equations)):
    eq = st.text_input(f"✏️ Equation {i+1}", value="0 0 0 0", help="Enter coefficients and constant separated by spaces")
    equations.append(eq)

if st.button("✅ Solve"):
    try:
        A = []
        b = []
        for eq in equations:
            coeffs = list(map(float, eq.split()))
            if len(coeffs) != num_equations + 1:
                st.error(f"⚠️ Equation {eq} does not have the correct number of terms. Please check your input.")
                st.stop()
            
            A.append(coeffs[:-1])
            b.append(coeffs[-1])
        
        A = np.array(A, dtype=float)
        b = np.array(b, dtype=float)
        
        solution = gaussian_elimination_pivoting(A, b)
        
        if solution is not None:
            st.success("✅ Solution Found!")
            with st.container():
                st.markdown('<div class="solution-box">', unsafe_allow_html=True)
                st.subheader("📌 Solution:")
                for i, val in enumerate(solution):
                    st.write(f"**x{i+1} = {val:.6f}**")
                st.markdown('</div>', unsafe_allow_html=True)

            # Verification
            st.subheader("🛠️ Verification:")
            for i, eq in enumerate(equations):
                coeffs = list(map(float, eq.split()))
                result = sum(c * x for c, x in zip(coeffs[:-1], solution))
                st.write(f"✅ **Equation {i+1}: {result:.6f} ≈ {coeffs[-1]}**")
    
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")

# Sidebar Guide with Icons
st.sidebar.markdown("""
## 📝 How to use:
1. **Enter the number of equations.**
2. **Input each equation** with coefficients and constant separated by spaces.  
   Example: `2 3 -1 5` for `2x + 3y - z = 5`
3. Click **"Solve"** to compute the solution instantly.
""")
