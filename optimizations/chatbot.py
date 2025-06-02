import os
import pandas as pd
import numpy as np
import firebase_admin
import re
from firebase_admin import credentials, db
from flask import Flask, request, jsonify, send_from_directory
from gtts import gTTS
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, List, Union, Tuple
from flask_cors import CORS

class IntegratedPostOfficeChatbot:
    def __init__(self, credentials_path: str, database_url: str):
        try:
            # Initialize Firebase Admin
            cred = credentials.Certificate("credentials.json")
            firebase_admin.initialize_app(cred, {
                'databaseURL': 'https://sih2024-559e6-default-rtdb.firebaseio.com/'
            })

            self.ref = db.reference('/postOffices')
            self.postoffices = self.ref.get()

            self.df = self._prepare_dataframe()
        except Exception as e:
            raise ValueError(f"Initialization error: {str(e)}")

    def _prepare_dataframe(self) -> pd.DataFrame:
        
        if not self.postoffices:
            raise ValueError('No data found in the database.')
        
        records = [
            {**details, 'id': postoffice_id} 
            for postoffice_id, details in self.postoffices.items()
        ]
        df = pd.DataFrame(records)
        
        numeric_columns = ['compliance', 'alert', 'garbageDetected']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        
        return df

    def handle_query(self, user_message: str) -> Dict[str, Any]:
        user_message = user_message.lower()

        # Advanced Analysis Queries
        analysis_keywords = {
            "compare": ["compare", "comparison", "versus", "vs"],
            "statistical": ["statistical", "statistics", "correlation", "distribution", "outliers"],
            "advanced": ["advanced analysis", "deep analysis", "comprehensive analysis"]
        }

        # Check for advanced analysis queries
        for analysis_type, keywords in analysis_keywords.items():
            if any(keyword in user_message for keyword in keywords):
                return self._handle_advanced_analysis(user_message, analysis_type)

        # Basic chatbot query 
        specific_address_result = self._check_specific_address(user_message)
        if specific_address_result:
            return {'reply': specific_address_result}

        # Compliance-related queries
        compliance_result = self._handle_compliance_query(user_message)
        if compliance_result:
            return {'reply': compliance_result}

        # Anomaly detection
        if any(keyword in user_message for keyword in ["anomalies", "anamoly", "anomaly detection", "detect anomalies"]):
            return {'reply': self._detect_anomalies()}

        # Garbage detection summary
        if any(keyword in user_message for keyword in ["garbage detection", "garbage detected", "garbage detect", "trash detection", "trash detected", "trash detect"]):
            return {'reply': self._get_garbage_detection_summary()}

        # General summary
        if any(keyword in user_message for keyword in ["summary", "summarize", "summarized", "summarizing"]):
            user_message_lower = user_message.lower()

            specific_address_result = self._check_specific_address(user_message)
            if specific_address_result:
                return {'reply': specific_address_result}

            else:
                return {'reply': self._generate_overall_summary()}

        # Alerts query
        if any(keyword in user_message for keyword in ["alerts", "warnings", "alert", "warning"]):
            return {'reply': self._get_alerts_summary()}

        return {'reply': "I'm sorry, I couldn't understand your specific query. Could you please rephrase?"}

    ####

    def _check_specific_address(self, user_message: str) -> str:
      
        for postoffice_id, details in self.postoffices.items():
            if details.get('name', '').lower() == user_message:
                return self._format_address_details(details)
        return None

    def _format_address_details(self, details: Dict[str, Any]) -> str:
       
        response_details = {
            'City': details.get('city', 'N/A'),
            'District': details.get('district', 'N/A'),
            'Email': details.get('email', 'N/A'),
            'Compliance Score': details.get('compliance', 'N/A'),
            'Garbage Detection': details.get('garbageDetectionData', {}).get('2024-12-06', {}).get('detections', 'N/A'),
            'Garbage Types': self._extract_garbage_types(details)
        }
        
        return '<br/>'.join(f"{k}: {v}" for k, v in response_details.items())

    def _extract_garbage_types(self, details: Dict[str, Any]) -> List[str]:
       
        garbage_types = details.get('garbageTypeData', {})
        if isinstance(garbage_types, dict):
            return [
                f"{type_data.get('type')}: {type_data.get('frequency')}"
                for type_data in garbage_types.values() 
                if isinstance(type_data, dict)
            ]
        elif isinstance(garbage_types, list):
            return [
                f"{type_data.get('type')}: {type_data.get('frequency')}"
                for type_data in garbage_types
            ]
        return []

    def _handle_compliance_query(self, user_message: str) -> str:
        
        if "compliance" not in self.df.columns:
            return None

        words1 = ["best compliance", "best performance", "top compliance", "best cleanliness"]
        words2 = ["worst compliance", "worst performance", "least compliance", "least cleanliness"]
        words3 = ["top 5 best compliance", "top 5 best performance", "top 5 best clean"]
        words4 = ["top 5 worst compliance", "top 5 worst performance", "top 5 worst clean"]

        if any(keyword in user_message for keyword in words3):
            top_5_best = self.df.nlargest(5, 'compliance')
            return "Top 5 Best Compliance Details:<br/>" + "<br/>".join(
                f"Name:  {row['name']}, Email: {row['email']}, Compliance Score: {row['compliance']}"
                for _, row in top_5_best.iterrows()
            )

        elif any(keyword in user_message for keyword in words4):
            top_5_worst = self.df.nsmallest(5, 'compliance')
            return "Top 5 Worst Compliance Details:<br/>" + "<br/>".join(
                f"Name:  {row['name']}, Email: {row['email']}, Compliance Score: {row['compliance']}"
                for _, row in top_5_worst.iterrows()
            )


        elif any(keyword in user_message for keyword in words1):
            best_row = self.df.loc[self.df["compliance"].idxmax()]
            return "<br/>".join(f"{k}: {v}" for k, v in {
                "Name": best_row.get("name", "Unknown"),
                "Address": best_row.get("address", "Unknown"),
                "City": best_row.get("city", "Unknown"),
                "District": best_row.get("district", "Unknown"),
                "Email": best_row.get("email", "Unknown"),
                "Compliance Score": best_row["compliance"],
            }.items())

        elif any(keyword in user_message for keyword in words2):
            worst_row = self.df.loc[self.df["compliance"].idxmin()]
            return "<br/>".join(f"{k}: {v}" for k, v in {
                "Name": worst_row.get("name", "Unknown"),
                "Address": worst_row.get("address", "Unknown"),
                "City": worst_row.get("city", "Unknown"),
                "District": worst_row.get("district", "Unknown"),
                "Email": worst_row.get("email", "Unknown"),
                "Compliance Score": worst_row["compliance"],
            }.items())

        return None



    def _detect_anomalies(self) -> str:
      
        if 'compliance' not in self.df.columns:
            return "Cannot perform anomaly detection: No compliance data available."

        model = IsolationForest(contamination=0.1, random_state=42)
        self.df['anomaly_score'] = model.fit_predict(self.df[['compliance']])
        self.df['anomaly_label'] = self.df['anomaly_score'].apply(lambda x: 'Anomalous' if x == -1 else 'Normal')

        anomalies = self.df[self.df['anomaly_label'] == 'Anomalous']
        return f"Detected {len(anomalies)} anomalies in compliance data:<br/>" + "<br/>".join(
            f"Address: {row['address']}, Compliance Score: {row['compliance']}"
            for _, row in anomalies.iterrows()
        )

    def _get_garbage_detection_summary(self) -> str:

        summary = "Garbage Detection Summary:<br/>"
        for _, row in self.df.iterrows():
            name = row.get("name", "Unknown")
            garbage_data = row.get("garbageDetectionData", {})
            detections = sum(
                data["detections"] 
                for date, data in garbage_data.items() 
                if isinstance(data, dict)
            )
            summary += f"{name}: {detections} detections<br/>"
        return summary

    def _generate_overall_summary(self) -> str:
       
        if 'compliance' not in self.df.columns:
            return "Insufficient data for generating summary."

        compliance_avg = self.df["compliance"].mean()
        non_compliance_avg = 100 - compliance_avg
        max_compliance = self.df['compliance'].max()
        min_compliance = self.df['compliance'].min()

        return (
            f"Overall Division Summary:<br/>"
            f"Average Compliance Score: {compliance_avg:.2f}<br/>"
            f"Average Non-Compliance Score: {non_compliance_avg:.2f}<br/>"
            f"Highest Compliance Score: {max_compliance}<br/>"
            f"Lowest Compliance Score: {min_compliance}<br/>"
            f"Total Post Offices: {len(self.df)}"
        )

    def _get_alerts_summary(self) -> str:
      
        if 'alert' not in self.df.columns:
            return "No alerts data available."

        alert_sum = self.df["alert"].sum()
        return f"Total number of alerts generated across Division: {alert_sum}"


    ###

    def _handle_advanced_analysis(self, user_message: str, analysis_type: str) -> Dict[str, Any]:
    
        try:
            def extract_addresses(message: str) -> List[str]:
                words = message.split()
    
                filtered_addresses = [
                    word.strip() 
                    for word in words 
                    if word.strip().lower() not in {
                        'compare', 'comparison', 'vs', 'versus', 
                        'statistical', 'analysis', 'the', 'a', 'an'
                    } and len(word.strip()) > 1
                ]
    
                return filtered_addresses

                
            if analysis_type == "compare":
                addresses = extract_addresses(user_message)
                if len(addresses) >= 2:
                    return self.compare_post_offices(addresses)
                else:
                    return {"reply": "Please provide at least two post office addresses to compare."}

            elif analysis_type == "statistical":
                # Determine specific statistical analysis type
                if "correlation" in user_message:
                    return self.advanced_statistical_analysis('correlation')
                elif "distribution" in user_message:
                    return self.advanced_statistical_analysis('distribution')
                elif "outlier" in user_message:
                    return self.advanced_statistical_analysis('outlier_detection')
                else:
                    return self.advanced_statistical_analysis()

            else:
                return {"reply": "Advanced analysis type not recognized. Try 'compare' or 'statistical' analysis."}

        except Exception as e:
            return {"error": f"Analysis error: {str(e)}"}

    def compare_post_offices(self, addresses: List[str]) -> Dict[str, str]:

        if self.df['name'].isna().any():
            self.df = self.df.dropna(subset=['name'])

        def normalize_name(name):
            return name.lower().strip()

        normalized_addresses = [normalize_name(addr) for addr in addresses]
        normalized_names = self.df['name'].apply(normalize_name)


        selected_offices = self.df[normalized_names.isin(normalized_addresses)]

        if len(selected_offices) < 2:
            missing_offices = [
                addr for addr in normalized_addresses 
                if addr not in normalized_names.tolist()
            ]

        comparison = {
        "Comparison Details": "Post Offices Comparative Analysis" ,
        "Compliance Comparison": self._compare_compliance(selected_offices),
        "Garbage Detection Comparison": self._compare_garbage_detection(selected_offices),
        "Alerts Comparison": self._compare_alerts(selected_offices),
        "Detailed Metrics": selected_offices[['address', 'city', 'district', 'compliance', 'alert']].to_dict(orient='records')
        }
    
        formatted_response = "Comparative Analysis:<br/><br/>"
        for key, value in comparison.items():
            if key != "Detailed Metrics":
                formatted_response += f"{key}:<br/>"
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        formatted_response += f"  {subkey}: {subvalue}<br/>"
                else:
                    formatted_response += f"  {value}<br/>"
    
        return {"reply": formatted_response}

    def _compare_compliance(self, selected_offices):
      
        return {"Average Compliance": selected_offices['compliance'].mean()}

    def _compare_garbage_detection(self, selected_offices):

        specific_row = selected_offices['name']

        if specific_row.empty:
            return 0  

        garbage_data = specific_row.iloc[0].get("garbageDetectionData", {})
    
        if not isinstance(garbage_data, dict):
            return 0 
            
        total_detections = sum(
            data.get("detections", 0)
            for date, data in garbage_data.items()
            if isinstance(data, dict)
        )
        return total_detections

    def _compare_alerts(self, selected_offices):
        return  selected_offices['alert'].sum()

    def advanced_statistical_analysis(self, analysis_type: str = 'correlation') -> Dict[str, Any]:

        numeric_columns = ['compliance', 'alert']                   ####  'garbageDetected'
        analysis_df = self.df[numeric_columns].dropna()
        
        if analysis_type == 'correlation':
            correlation_matrix = analysis_df.corr()
 
            formatted_response = "Correlation Analysis:<br/>"
            for col1 in correlation_matrix.columns:
                for col2 in correlation_matrix.columns:
                    if col1 != col2:
                        correlation = correlation_matrix.loc[col1, col2]
                        formatted_response += f"{col1} vs {col2}: {correlation:.2f}<br/>"
            
            return {"reply": formatted_response}
        
        elif analysis_type == 'distribution':
            desc_stats = analysis_df.describe()
            
            formatted_response = "Distribution Analysis:<br/>"
            for col in desc_stats.columns:
                formatted_response += f"<br/>{col.upper()} Statistics:<br/>"
                for stat, value in desc_stats[col].items():
                    formatted_response += f"  {stat}: {value:.2f}<br/>"
            
            return {"reply": formatted_response}
        
        elif analysis_type == 'outlier_detection':
            outlier_columns = ['compliance', 'alert']
            X = analysis_df[outlier_columns]
            
            clf = IsolationForest(contamination=0.1, random_state=42)
            y_pred = clf.fit_predict(X)
            
            outliers = self.df[y_pred == -1]
            
            formatted_response = "Outlier Detection:<br/>"
            formatted_response += f"Total Outliers Detected: {len(outliers)}<br/><br/>"
            formatted_response += "Outlier Details:<br/>"
            for _, row in outliers.iterrows():
                formatted_response += (
                    f"Address: {row.get('address', 'Unknown')}<br/>"
                    f"  Compliance: {row.get('compliance', 'N/A')}<br/>"
                    f"  Alerts: {row.get('alerts', 'N/A')}<br/><br/>"
                )
            
            return {"reply": formatted_response}
        
        else:
            raise ValueError(f"Unsupported analysis type: {analysis_type}")

    def generate_voice_reply(self, text: str, user_id: str) -> str:
        """Generates a voice reply as an audio file."""
        try:
            # Create a unique filename for the user's reply
            filename = f"reply_{user_id}.mp3"
            filepath = os.path.join("static/audio", filename)

            # Use gTTS to generate the audio
            tts = gTTS(text=text, lang='en')
            tts.save(filepath)

            return f"/static/audio/{filename}"
        except Exception as e:
            raise ValueError(f"Error generating voice reply: {str(e)}")


def create_app(credentials_path: str, database_url: str) -> Flask:
    app = Flask(__name__)
    CORS(app)
    chatbot = IntegratedPostOfficeChatbot(credentials_path, database_url)

    if not os.path.exists("static/audio"):
        os.makedirs("static/audio")

    @app.route('/chat', methods=['POST'])
    def chat():
        user_message = request.json.get('message', '')

        user_id = request.json.get('user_id', 'default_user')  # Use unique user IDs if available
        
        if not user_message:
            return jsonify({'reply': 'Please provide a message.'})
        
        try:
            response = chatbot.handle_query(user_message)

            # Generate voice reply
            text_reply = response.get('reply', 'No reply available.')
            # voice_url = chatbot.generate_voice_reply(text_reply, user_id)

            return jsonify({'reply': text_reply})
        except Exception as e:
            return jsonify({'reply': f"Error: {str(e)}"}), 500


    return app

def main():

    CREDENTIALS_PATH = os.environ.get('FIREBASE_CREDENTIALS_PATH', "C:/Users/skmda/OneDrive/Desktop/credentials.json")
    DATABASE_URL = os.environ.get('FIREBASE_DATABASE_URL', 'https://sih2024-559e6-default-rtdb.firebaseio.com/')

    app = create_app(CREDENTIALS_PATH, DATABASE_URL)
    app.run(debug=True)

if __name__ == '__main__':
    main()
