import firebase_admin
from firebase_admin import credentials, db
from flask import Flask, request, jsonify
import pandas as pd
import openai
import os
from flask_cors import CORS

cred = credentials.Certificate("credentials.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://sih2024-559e6-default-rtdb.firebaseio.com/'
})

app = Flask(__name__)
CORS(app)

openai.api_key = "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"

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message', '')
    
    if not user_message:
        return jsonify({'reply': 'Please provide a message.'}), 400
    
    try:
        ref = db.reference('/postOffices')
        postoffices = ref.get()
        
        if not postoffices:
            return jsonify({'reply': 'No data found in the database.'}), 404
        
        records = [
            {**details, 'id': postoffice_id} 
            for postoffice_id, details in postoffices.items()
        ]
        df = pd.DataFrame(records)


        if "compliance" in df.columns:
            df["compliance"] = pd.to_numeric(df["compliance"], errors="coerce")
            
            best_compliance_row = df.loc[df["compliance"].idxmax()]
            worst_compliance_row = df.loc[df["compliance"].idxmin()]

            best_compliance_details = {
                "Address": best_compliance_row.get("address", "Unknown"),
                "City": best_compliance_row.get("city", "Unknown"),
                "District": best_compliance_row.get("district", "Unknown"),
                "Email": best_compliance_row.get("email", "Unknown"),
                "Compliance Score": best_compliance_row["compliance"],
            }

            worst_compliance_details = {
                "Address": worst_compliance_row.get("address", "Unknown"),
                "City": worst_compliance_row.get("city", "Unknown"),
                "District": worst_compliance_row.get("district", "Unknown"),
                "Email": worst_compliance_row.get("email", "Unknown"),
                "Compliance Score": worst_compliance_row["compliance"],
            }

            words1 = ["best compliance", "best performance", "top compliance", "best cleanliness"]
            words2 = ["worst compliance", "worst performance", "least compliance", "least cleanliness"]
            words3 = ["top 5 best compliance", "top 5 best performance", "top 5 best clean", "top 5 best cleanliness"]
            words4 = ["top 5 worst compliance", "top 5 worst performance", "top 5 worst clean", "top 5 worst cleanliness"]


            if any(keyword in user_message.lower() for keyword in words1) and "top 5" not in user_message.lower():
                summary = "Best Compliance:\n" + "\n".join(f"{k}: {v}" for k, v in best_compliance_details.items())
                return jsonify({'reply': summary})

            elif any(keyword in user_message.lower() for keyword in words2) and "top 5" not in user_message.lower():
                summary = "Worst Compliance:\n" + "\n".join(f"{k}: {v}" for k, v in worst_compliance_details.items())
                return jsonify({'reply': summary})
            
            elif any(keyword in user_message.lower() for keyword in words3):
                top_5_best = df.nlargest(5, 'compliance')[['address', 'city', 'district', 'email', 'compliance']]
                summary = "Top 5 Best Compliance Addresses:\n"
                for _, row in top_5_best.iterrows():
                    summary += f"Address: {row['address']}, City: {row['city']}, District: {row['district']}, Email: {row['email']}, Compliance Score: {row['compliance']}\n"
                return jsonify({'reply': summary})

            elif any(keyword in user_message.lower() for keyword in words4):
                top_5_worst = df.nsmallest(5, 'compliance')[['address', 'city', 'district', 'email', 'compliance']]
                summary = "Top 5 Worst Compliance Addresses:\n"
                for _, row in top_5_worst.iterrows():
                    summary += f"Address: {row['address']}, City: {row['city']}, District: {row['district']}, Email: {row['email']}, Compliance Score: {row['compliance']}\n"
                return jsonify({'reply': summary})


        for postoffice_id, details in postoffices.items():
            if details.get('address', '').lower() == user_message.lower():
                response_details = {
                    'City': details.get('city', 'N/A'),
                    'District': details.get('district', 'N/A'),
                    'Email': details.get('email', 'N/A'),
                    'Compliance Score': details.get('compliance', 'N/A'),
                    'Garbage Detection': details.get('garbageDetectionData', {}).get('2024-12-06', {}).get('detections', 'N/A'),
                    'Garbage Types': []
                }
                
                garbage_types = details.get('garbageTypeData', {})
                if isinstance(garbage_types, dict):
                    response_details['Garbage Types'] = [
                        f"{type_data.get('type')}: {type_data.get('frequency')}"
                        for type_data in garbage_types.values() 
                        if isinstance(type_data, dict)
                    ]
                elif isinstance(garbage_types, list):
                    response_details['Garbage Types'] = [
                        f"{type_data.get('type')}: {type_data.get('frequency')}"
                        for type_data in garbage_types
                    ]
                
                return jsonify({'reply': '\n'.join(f"{k}: {v}" for k, v in response_details.items())})

        
        keywords1 = ["summary", "summarize", "summarized", "summarizing"]
        keywords2 = ["garbage detection", "garbage detected", "garbage detect", "trash detection", "trash detected", "trash detect"]
        keywords3 = ["alerts", "warnings", "alert", "warning"]

        if any(keyword in user_message.lower() for keyword in keywords3):
            alert_sum = df["alerts"].sum()
            summary = (
                f"Total no of alerts generated across Division: {alert_sum}. "
            )

        if any(keyword in user_message.lower() for keyword in keywords1):
            compliance_avg = df["compliance"].mean()

            df["nonCompliance"] = 100 - df["compliance"]
            non_compliance_avg = df["nonCompliance"].mean()

            summary = (
                f"Average compliance score across Division: {compliance_avg:.2f}. \n "
                f"Average non-compliance score across Division: {non_compliance_avg:.2f}."
            )

        elif any(keyword in user_message.lower() for keyword in keywords2):
            summary = "Garbage Detection Summary: \n"
            for _, row in df.iterrows():
                name = row.get("name", "Unknown")
                garbage_data = row.get("garbageDetectionData", {})
                detections = sum(
                    data["detections"] 
                    for date, data in garbage_data.items() 
                    if isinstance(data, dict)
                )
                summary += f"{name}: {detections} detections\n"

        else:

            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant summarizing Firebase data."},
                    {"role": "user", "content": user_message}
                ]
            )
            summary = response['choices'][0]['message']['content']
        
        return jsonify({'reply': summary})
    
    except Exception as e:
        return jsonify({'reply': f"Error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)



