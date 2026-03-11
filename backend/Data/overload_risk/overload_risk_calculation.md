PROCEDURE: Overload Risk Classification Data preprocessing and load percentage calculation steps.

1. LOAD DATA:
   IMPORT "master_data.xlsx" as a DataFrame.

2. DATA CLEANING:
   REMOVE rows containing Null (NaN) values in current columns (I_A, I_B, I_C).
   REMOVE rows where current values <= 0 (Filter sensor noise/zero-load errors).

3. ESTABLISH RATED CAPACITY (I_rated):
   CALCULATE the 98th Percentile of all current readings.
   SET this value as the Equipment Operational Limit (Ref: IEEE 141).

4. APPLY LAW OF LOAD UTILIZATION:
   FOR each row in Dataset:
     a. FIND the Maximum Current among Phases A, B, and C.
     b. CALCULATE Load Percentage = (Max Current / I_rated) * 100.
     c. IDENTIFY Peak Phase = Column Name of Max Current.

5. CLASSIFY OVERLOAD RISK (Ref: NFPA 70 / NEC):
   IF Load Percentage < 80%:
     SET Risk Level = "Low"
   ELSE IF Load Percentage < 90%:
     SET Risk Level = "Medium"
   ELSE:
     SET Risk Level = "High"

6. EXPORT:
   SAVE processed data to "equipment_overload_analysis.xlsx".