import numpy as np
from agentdriver.planning.qwen_compatibility import parse_qwen_trajectory

response = """
**Trajectory Waypoints:**
1. **0.5s:** $(-0.02, 2.16)$
2. **1.0s:** $(-0.03, 4.30)$
3. **1.5s:** $(-0.03, 6.43)$
4. **2.0s:** $(-0.02, 8.54)$
5. **2.5s:** $(0.00, 10.64)$
6. **3.0s:** $(0.03, 12.72)$
"""

expected = np.array([
    [-0.02, 2.16],
    [-0.03, 4.30],
    [-0.03, 6.43],
    [-0.02, 8.54],
    [0.00, 10.64],
    [0.03, 12.72],
])

actual = parse_qwen_trajectory(response)

np.testing.assert_allclose(actual, expected)

print("PASS: Qwen trajectory correctly parsed")
print(actual)



# Scenario 2: Timestamped waypoints using "seconds" and (x, y) notation
# Previously failed during the 2-scenario AgentDriver experiment.
response_2 = """
The trajectory of the vehicle, calculated using the provided
velocity and acceleration values, results in the following waypoints:

- **0.5 seconds**:
  $ (x, y) = (0.028, 1.13) $
- **1.0 seconds**:
  $ (x, y) = (0.040, 2.20) $
- **1.5 seconds**:
  $ (x, y) = (0.038, 3.21) $
- **2.0 seconds**:
  $ (x, y) = (0.020, 4.16) $
- **2.5 seconds**:
  $ (x, y) = (-0.013, 5.05) $
- **3.0 seconds**:
  $ (x, y) = (-0.060, 5.88) $

### Key Considerations:
The vehicle remains within the drivable area.
"""

expected_2 = np.array([
    [0.028, 1.13],
    [0.040, 2.20],
    [0.038, 3.21],
    [0.020, 4.16],
    [-0.013, 5.05],
    [-0.060, 5.88],
])

np.testing.assert_allclose(
    parse_qwen_trajectory(response_2), expected_2
)

print("PASS: Scenario 2 (previously failed)")


# Scenario 4: Numbered waypoints with timing context
response_4 = """
The trajectory of the vehicle, calculated based on the given initial
velocity and acceleration, results in the following waypoints at
0.5s, 1.0s, 1.5s, 2.0s, 2.5s, and 3.0s intervals:

**Waypoints:**
1. **(-0.02, 2.16)**
2. **(-0.03, 4.30)**
3. **(-0.03, 6.43)**
4. **(-0.02, 8.54)**
5. **(0.00, 10.64)**
6. **(0.03, 12.72)**

### Key Observations:
The vehicle continues moving forward.
"""

expected_4 = np.array([
    [-0.02, 2.16],
    [-0.03, 4.30],
    [-0.03, 6.43],
    [-0.02, 8.54],
    [0.00, 10.64],
    [0.03, 12.72],
])

np.testing.assert_allclose(
    parse_qwen_trajectory(response_4), expected_4
)

print("PASS: Scenario 4 (numbered waypoints)")

# Scenario 5: Numbered waypoints with calculations
response_5 = """
1. **$ t = 0.5 $ s**
   **Waypoint**: $ (0.0025, 0.3775) $

2. **$ t = 1.0 $ s**
   **Waypoint**: $ (0.0, 0.65) $

3. **$ t = 1.5 $ s**
   **Waypoint**: $ (0.0, 0.8175) $

4. **$ t = 2.0 $ s**
   **Waypoint**: $ (0.0, 0.88) $

5. **$ t = 2.5 $ s**
   **Waypoint**: $ (0.0, 0.8375) $

6. **$ t = 3.0 $ s**
   **Waypoint**: $ (0.0, 0.69) $
"""

expected_5 = np.array([
    [0.0025, 0.3775],
    [0.0, 0.65],
    [0.0, 0.8175],
    [0.0, 0.88],
    [0.0, 0.8375],
    [0.0, 0.69],
])

np.testing.assert_allclose(
    parse_qwen_trajectory(response_5), expected_5
)
print("PASS: Scenario 5")


# Scenario 5: Memory-free baseline with calculations between timestamps
# Token: fc03376a8e4f4ee4830756bb3e212bd8
# Previously failed because the waypoint was several lines after its timestamp.

response_no_memory_5 = r"""
To ensure the vehicle remains within the road and avoids the left
shoulder, the trajectory must be adjusted to keep $ x \geq 0 $.

### Adjusted Waypoints (Ensuring $ x \geq 0 $):

1. **$ t = 0.5 $ s**
   $ x = 0.0025 $ m (safe, as it is positive)
   $ y = 0.86 \times 0.5 + 0.5 \times (-0.42) \times (0.5)^2 = 0.43 - 0.0525 = 0.3775 $ m
   **Waypoint**: $ (0.0025, 0.3775) $

2. **$ t = 1.0 $ s**
   $ x = 0.0 $ m (edge of the road, acceptable)
   $ y = 0.86 \times 1.0 + 0.5 \times (-0.42) \times (1.0)^2 = 0.86 - 0.21 = 0.65 $ m
   **Waypoint**: $ (0.0, 0.65) $

3. **$ t = 1.5 $ s**
   $ x = 0.0 $ m (stay on the edge)
   $ y = 0.86 \times 1.5 + 0.5 \times (-0.42) \times (1.5)^2 = 1.29 - 0.4725 = 0.8175 $ m
   **Waypoint**: $ (0.0, 0.8175) $

4. **$ t = 2.0 $ s**
   $ x = 0.0 $ m
   $ y = 0.86 \times 2.0 + 0.5 \times (-0.42) \times (2.0)^2 = 1.72 - 0.84 = 0.88 $ m
   **Waypoint**: $ (0.0, 0.88) $

5. **$ t = 2.5 $ s**
   $ x = 0.0 $ m
   $ y = 0.86 \times 2.5 + 0.5 \times (-0.42) \times (2.5)^2 = 2.15 - 1.3125 = 0.8375 $ m
   **Waypoint**: $ (0.0, 0.8375) $

6. **$ t = 3.0 $ s**
   $ x = 0.0 $ m
   $ y = 0.86 \times 3.0 + 0.5 \times (-0.42) \times (3.0)^2 = 2.58 - 1.89 = 0.69 $ m
   **Waypoint**: $ (0.0, 0.69) $

### Key Adjustments:
The vehicle avoids negative x-values while maintaining forward motion.
"""

expected_no_memory_5 = np.array([
    [0.0025, 0.3775],
    [0.0, 0.65],
    [0.0, 0.8175],
    [0.0, 0.88],
    [0.0, 0.8375],
    [0.0, 0.69],
])

np.testing.assert_allclose(
    parse_qwen_trajectory(response_no_memory_5),
    expected_no_memory_5
)

print("PASS: Memory-free Scenario 5 (calculation-heavy waypoints)")

# Scenario 6: Numbered waypoints with timing explained in the rationale
response_6 = """
**Planned Trajectory Waypoints:**

1. **(-0.02, 1.51)**
2. **(-0.07, 3.02)**
3. **(-0.17, 4.53)**
4. **(-0.30, 6.04)**
5. **(-0.48, 7.55)**
6. **(-0.69, 9.06)**

---

**Rationale:**
- The trajectory is calculated using the current velocity
  **(0.01, 3.02)** and acceleration **(-0.16, 0)**,
  integrating over time intervals (0.5s, 1.0s, etc.).
- The x-values decrease slightly due to acceleration.
"""

expected_6 = np.array([
    [-0.02, 1.51],
    [-0.07, 3.02],
    [-0.17, 4.53],
    [-0.30, 6.04],
    [-0.48, 7.55],
    [-0.69, 9.06],
])

np.testing.assert_allclose(
    parse_qwen_trajectory(response_6), expected_6
)

print("PASS: Scenario 6 (numbered waypoints)")

# Scenario 12: Markdown coordinates and LaTeX
response_12 = """
- **0.5s**: (0.005, 2.575)
- **1.0s**: (0.01, 5.15)
- **1.5s**: (0.015, 7.725)
- **2.0s**: (0.02, 10.3)
- **2.5s**: (0.025, 12.875)
- **3.0s**: (0.03, 15.45)

Final Answer:
\\boxed{(0.005, 2.575), (0.01, 5.15)}
"""

expected_12 = np.array([
    [0.005, 2.575],
    [0.01, 5.15],
    [0.015, 7.725],
    [0.02, 10.3],
    [0.025, 12.875],
    [0.03, 15.45],
])

np.testing.assert_allclose(
    parse_qwen_trajectory(response_12), expected_12
)
print("PASS: Scenario 12")


# Scenario 20: Separate position entries
response_20 = """
1. **t = 0.5s:**
   - **Position:** (0.004, 1.42)

2. **t = 1.0s:**
   - **Position:** (0.005, 2.58)

3. **t = 1.5s:**
   - **Position:** (0.004, 3.48)

4. **t = 2.0s:**
   - **Position:** (0.000, 4.12)

5. **t = 2.5s:**
   - **Position:** (-0.006, 4.51)

6. **t = 3.0s:**
   - **Position:** (-0.015, 4.64)
"""

expected_20 = np.array([
    [0.004, 1.42],
    [0.005, 2.58],
    [0.004, 3.48],
    [0.000, 4.12],
    [-0.006, 4.51],
    [-0.015, 4.64],
])

np.testing.assert_allclose(
    parse_qwen_trajectory(response_20), expected_20
)
print("PASS: Scenario 20")



# Scenario 2: LaTeX spacing in Qwen coordinates
response_latex = r"""
The trajectory of the vehicle is calculated as follows:

### Waypoints:
1. **0.5 seconds:**
   $ (0.03, \ 1.13) $
2. **1.0 seconds:**
   $ (0.04, \ 2.20) $
3. **1.5 seconds:**
   $ (0.04, \ 3.21) $
4. **2.0 seconds:**
   $ (0.02, \ 4.16) $
5. **2.5 seconds:**
   $ (-0.01, \ 5.05) $
6. **3.0 seconds:**
   $ (-0.06, \ 5.88) $

### Final Answer:
\boxed{
\begin{aligned}
&\text{0.5s: } (0.03, \ 1.13), \\
&\text{1.0s: } (0.04, \ 2.20)
\end{aligned}
}
"""

expected_latex = np.array([
    [0.03, 1.13],
    [0.04, 2.20],
    [0.04, 3.21],
    [0.02, 4.16],
    [-0.01, 5.05],
    [-0.06, 5.88],
])

np.testing.assert_allclose(
    parse_qwen_trajectory(response_latex),
    expected_latex
)

print("PASS: Scenario 2 (LaTeX spacing)")


# Scenario 20: Separate x/y coordinates using LaTeX \approx
# Token: 9c9cdfaf964f4776b30f1edbdf9adb0b
# Previously failed during the 20-scenario AgentDriver experiment.

response_20_latex = r"""
The vehicle's trajectory, based on the given initial velocity and acceleration, results in the following waypoints at 0.5-second intervals over 3 seconds:

- **At 0.5 seconds**:
  $ x \approx 0.004 \, \text{m}, \, y \approx 1.45 \, \text{m} $
- **At 1.0 seconds**:
  $ x \approx 0.005 \, \text{m}, \, y \approx 2.70 \, \text{m} $
- **At 1.5 seconds**:
  $ x \approx 0.006 \, \text{m}, \, y \approx 3.74 \, \text{m} $
- **At 2.0 seconds**:
  $ x \approx 0.007 \, \text{m}, \, y \approx 4.60 \, \text{m} $
- **At 2.5 seconds**:
  $ x \approx 0.008 \, \text{m}, \, y \approx 5.26 \, \text{m} $
- **At 3.0 seconds**:
  $ x \approx 0.009 \, \text{m}, \, y \approx 5.72 \, \text{m} $

These waypoints reflect a smooth deceleration in the y-direction.

**Final Answer:**
$$
\boxed{
\begin{aligned}
&\text{Waypoints (x, y) at 0.5s intervals:} \\
&\quad (0.004, 1.45), \, (0.005, 2.70), \, (0.006, 3.74), \, (0.007, 4.60), \, (0.008, 5.26), \, (0.009, 5.72)
\end{aligned}
}
$$
"""

expected_20_latex = np.array([
    [0.004, 1.45],
    [0.005, 2.70],
    [0.006, 3.74],
    [0.007, 4.60],
    [0.008, 5.26],
    [0.009, 5.72],
])

np.testing.assert_allclose(
    parse_qwen_trajectory(response_20_latex),
    expected_20_latex
)

print("PASS: Scenario 20 (separate LaTeX x/y coordinates)")


# Regression: final scenario of the memory-free 20-token experiment.
# Token: 9c9cdfaf964f4776b30f1edbdf9adb0b
# Explicit Position labels, plus a later explanatory bullet with "t = 3.0s".
response_no_memory_final = """
The vehicle's trajectory, based on the assumption that the current position is at (0, 0) and the goal is to decelerate to a stop, is as follows:

### **Waypoints (x, y) at 0.5s intervals:**
1. **t = 0.5s:**
   - **x ≈ 0.004 m**
   - **y ≈ 1.42 m**
   - **Position:** (0.004, 1.42)

2. **t = 1.0s:**
   - **x ≈ 0.005 m**
   - **y ≈ 2.58 m**
   - **Position:** (0.005, 2.58)

3. **t = 1.5s:**
   - **x ≈ 0.004 m**
   - **y ≈ 3.48 m**
   - **Position:** (0.004, 3.48)

4. **t = 2.0s:**
   - **x ≈ 0.000 m**
   - **y ≈ 4.12 m**
   - **Position:** (0.000, 4.12)

5. **t = 2.5s:**
   - **x ≈ -0.006 m**
   - **y ≈ 4.51 m**
   - **Position:** (-0.006, 4.51)

6. **t = 3.0s:**
   - **x ≈ -0.015 m**
   - **y ≈ 4.64 m**
   - **Position:** (-0.015, 4.64)

---

### **Key Notes:**
- **Deceleration Assumption:** The y-component of acceleration is adjusted to ensure the vehicle stops at **t = 3.0s** (i.e., final velocity in y = 0 m/s). This results in a deceleration of **-1.03 m/s²** in the y-direction.
- **X-Coordinate:** The x-component of velocity is negligible (0.01 m/s), so the x-coordinate changes minimally over time.
- **Safety Margin:** The vehicle stops at **y ≈ 4.64 m**, which is far from the pedestrians (assumed to be at **y ≈ 29.17 m**), ensuring a safe trajectory.

---

### **Conclusion:**
The waypoints are calculated to ensure the vehicle decelerates smoothly to a stop, avoiding any collision with pedestrians. The trajectory is safe and adheres to the given constraints.
"""

expected_no_memory_final = np.array([
    [0.004, 1.42],
    [0.005, 2.58],
    [0.004, 3.48],
    [0.000, 4.12],
    [-0.006, 4.51],
    [-0.015, 4.64],
])

np.testing.assert_allclose(
    parse_qwen_trajectory(response_no_memory_final),
    expected_no_memory_final,
)
print("PASS: Memory-free final scenario (Position bullets and explanatory t=3.0s)")
