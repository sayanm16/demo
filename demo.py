import streamlit as st
import numpy as np
import time

# Custom CSS for animations & effects
st.markdown("""
    <style>
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }

        .stApp {
            animation: fadeIn 1.5s ease-in-out;
            background: linear-gradient(to right, #74ebd5, #acb6e5);
        }

        .title {
            font-size: 30px;
            text-align: center;
            font-weight: bold;
            color: #ffffff;
            animation: fadeIn 2s;
        }

        .stButton>button {
            background-color: #007bff;
            color: white;
            font-size: 16px;
            border-radius: 8px;
            width: 100%;
            transition: all 0.3s ease-in-out;
            animation: fadeIn 2s;
        }

        .stButton>button:hover {
            background-color: #0056b3;
            transform: scale(1.05);
        }

        .solution-box {
            padding: 15px;
            border-radius: 10px;
            background: white;
            box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
            margin-top: 20px;
            animation: fadeIn 2s ease-in-out;
        }
    </style>
""", unsafe_allow_html=True)

# Title with fade-in effect
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

if st.button("🚀 Solve with Animation"):
    with st.spinner("🛠️ Solving... Please wait!"):
        time.sleep(2)  # Simulate computation time
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
                
                # Show progress bar for effect
                progress_bar = st.progress(0)
                for percent_complete in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(percent_complete + 1)
                
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
3. Click **"Solve with Animation"** to compute the solution.
4. Watch the progress bar and enjoy the smooth animation effects! 🎉
""")
