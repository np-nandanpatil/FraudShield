import os
import sys
import csv
import json
import random
import uuid
from datetime import datetime, timedelta
from functools import wraps

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, request, jsonify, render_template, session, redirect, url_for, flash
from src.utils.predictor import RealTimePredictor

app = Flask(__name__)
app.secret_key = os.urandom(24)  # For session management

# Load admin password from environment variable or use default for demo
ADMIN_PASSWORD = os.environ.get('SYNTHHACK_ADMIN_PASSWORD', 'synthhack_admin')

# Initialize the fraud detection system
predictor = RealTimePredictor('models/fraud_detection_model.pkl', 'models/feature_engineer.pkl')

# In-memory storage for transactions and OTPs
transactions_db = {}
otps = {}

# Add simple authentication decorator
def admin_required(f):
    """Decorator to require admin password for dashboard access"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'is_admin' not in session or not session['is_admin']:
            return redirect(url_for('admin_login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
@admin_required
def dashboard():
    """Payment gateway dashboard showing transaction history"""
    txn_data = []
    try:
        with open('data/processed_transactions.csv', 'r') as file:
            reader = csv.DictReader(file)
            txn_data = list(reader)
            
            # Calculate statistics
            total_transactions = len(txn_data)
            fraud_transactions = [t for t in txn_data if t['Is_Fraud'].lower() == 'true']
            fraud_count = len(fraud_transactions)
            legitimate_count = total_transactions - fraud_count
            fraud_rate = (fraud_count / total_transactions * 100) if total_transactions > 0 else 0
            
            # Calculate fraud types distribution
            fraud_types = {}
            for t in fraud_transactions:
                fraud_type = t['Fraud_Type']
                if fraud_type and fraud_type != '-':
                    fraud_types[fraud_type] = fraud_types.get(fraud_type, 0) + 1
            
            # Calculate transaction volume over time
            transactions_by_date = {}
            for t in txn_data:
                date = t['Timestamp'].split('T')[0]  # Get just the date part
                amount = float(t['Amount'])
                if date in transactions_by_date:
                    transactions_by_date[date]['volume'] += amount
                    transactions_by_date[date]['count'] += 1
                else:
                    transactions_by_date[date] = {'volume': amount, 'count': 1}
            
            # Convert to sorted lists for the chart
            dates = sorted(transactions_by_date.keys())
            volumes = [transactions_by_date[date]['volume'] for date in dates]
            
    except FileNotFoundError:
        total_transactions = 0
        fraud_count = 0
        legitimate_count = 0
        fraud_rate = 0
        fraud_types = {}
        dates = []
        volumes = []
        
    return render_template('dashboard.html',
                         transactions=txn_data,
                         stats={
                             'total_transactions': total_transactions,
                             'fraud_count': fraud_count,
                             'legitimate_count': legitimate_count,
                             'fraud_rate': round(fraud_rate, 1),
                             'fraud_types': fraud_types,
                             'transaction_volume': {
                                 'dates': dates,
                                 'volumes': volumes
                             }
                         })

# Ensure the CSV file exists with headers
def ensure_transaction_csv_exists():
    csv_path = 'data/processed_transactions.csv'
    if not os.path.exists(csv_path):
        app.logger.info(f"Creating new transaction CSV file at {csv_path}")
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                'Transaction_ID',
                'Timestamp',
                'Amount',
                'Sender_ID',
                'Receiver_ID',
                'Transaction_Type',
                'Merchant_Type',
                'Device_ID',
                'Location',
                'Is_Fraud',
                'Fraud_Type',
                'Fraud_Probability',
                'Status'
            ])
        # Return True if we created a new file
        return True
    # Return False if file already existed
    return False

@app.route('/payment_methods')
def payment_methods():
    """Display available payment methods"""
    return render_template('payment_methods.html')

@app.route('/netbanking')
def netbanking():
    """Display netbanking options"""
    banks = [
        {"id": "sbi", "name": "State Bank of India"},
        {"id": "hdfc", "name": "HDFC Bank"},
        {"id": "icici", "name": "ICICI Bank"},
        {"id": "axis", "name": "Axis Bank"},
        {"id": "kotak", "name": "Kotak Mahindra Bank"},
        {"id": "pnb", "name": "Punjab National Bank"}
    ]
    return render_template('netbanking.html', banks=banks)

@app.route('/wallets')
def wallets():
    """Display wallet options"""
    wallets = [
        {"id": "paytm", "name": "Paytm"},
        {"id": "phonepe", "name": "PhonePe"},
        {"id": "gpay", "name": "Google Pay"},
        {"id": "amazonpay", "name": "Amazon Pay"},
        {"id": "mobikwik", "name": "MobiKwik"}
    ]
    return render_template('wallets.html', wallets=wallets)

@app.route('/process_transaction', methods=['POST'])
def process_transaction():
    try:
        transaction_data = request.json
        
        # Input validation
        if not transaction_data:
            return jsonify({
                'status': 'error',
                'message': 'No transaction data provided'
            }), 400
        
        # Validate required fields
        required_fields = ['Amount', 'Transaction_Type']
        missing_fields = [field for field in required_fields if field not in transaction_data]
        if missing_fields:
            return jsonify({
                'status': 'error',
                'message': f'Missing required fields: {", ".join(missing_fields)}'
            }), 400
        
        # Validate amount is a number
        try:
            amount = float(transaction_data.get('Amount', 0))
            if amount <= 0:
                return jsonify({
                    'status': 'error',
                    'message': 'Amount must be greater than zero'
                }), 400
            # Update with validated amount
            transaction_data['Amount'] = amount
        except (ValueError, TypeError):
            return jsonify({
                'status': 'error',
                'message': 'Invalid amount format'
            }), 400
            
        # Generate a unique transaction ID if not provided
        if 'Transaction_ID' not in transaction_data:
            transaction_data['Transaction_ID'] = f"TXN_{uuid.uuid4().hex[:10]}"
        
        # Add Receiver_ID if not present
        if 'Receiver_ID' not in transaction_data:
            transaction_data['Receiver_ID'] = 'Merchant_' + str(hash(transaction_data.get('Merchant_Type', '')) % 10000)
        
        # Step 1: Fraud detection
        result = predictor.predict_transaction(transaction_data)
        
        # If fraudulent, block immediately
        if result['is_fraud']:
            log_transaction(transaction_data, result, "BLOCKED")
            return jsonify({
                'status': 'blocked',
                'reason': result['fraud_type'],
                'confidence': result['fraud_probability']
            }), 403
        
        # Step 2: For legitimate transactions, proceed with payment flow
        txn_id = transaction_data['Transaction_ID']
        
        # Store transaction for later verification
        transactions_db[txn_id] = {
            'data': transaction_data,
            'result': result,
            'status': 'PENDING'
        }
        
        # Generate OTP for authentication (simulate bank OTP)
        otp = str(random.randint(100000, 999999))
        otps[txn_id] = otp
        
        # In real implementation, OTP would be sent via SMS/email
        
        return jsonify({
            'status': 'pending_verification',
            'transaction_id': txn_id,
            'redirect_url': f"/verify_otp/{txn_id}",
            'confidence': result['fraud_probability']
        }), 200
    except Exception as e:
        app.logger.error(f"Error processing transaction: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': 'An unexpected error occurred while processing the transaction'
        }), 500

@app.route('/verify_otp/<txn_id>', methods=['GET', 'POST'])
def verify_otp(txn_id):
    """OTP verification page"""
    if request.method == 'POST':
        user_otp = request.form.get('otp')
        
        # Verify OTP
        if txn_id in otps and user_otp == otps[txn_id]:
            # OTP verified, complete transaction
            transaction = transactions_db.get(txn_id)
            if transaction:
                # Update status
                transaction['status'] = 'COMPLETED'
                log_transaction(transaction['data'], transaction['result'], "COMPLETED")
                
                # Clean up
                del otps[txn_id]
                
                return render_template('success.html', transaction=transaction)
        
        # Invalid OTP
        return render_template('verify_otp.html', txn_id=txn_id, error="Invalid OTP. Please try again.")
    
    # GET request - show OTP form
    return render_template('verify_otp.html', txn_id=txn_id)

@app.route('/payment_status/<txn_id>')
def payment_status(txn_id):
    """Check payment status"""
    transaction = transactions_db.get(txn_id)
    if not transaction:
        return jsonify({'status': 'not_found'}), 404
    
    return jsonify({
        'status': transaction['status'],
        'transaction_id': txn_id,
        'amount': transaction['data'].get('Amount'),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/transactions', methods=['GET'])
@admin_required
def api_transactions():
    """API endpoint for transactions data"""
    txn_data = []
    try:
        app.logger.info("Loading transactions from CSV file")
        csv_path = 'data/processed_transactions.csv'
        
        # Check if file exists first
        if not os.path.exists(csv_path):
            app.logger.warning(f"Transaction CSV file not found at {csv_path}, creating new one")
            ensure_transaction_csv_exists()
            return jsonify([])  # Return empty array if no transactions
            
        # Try to read the file
        with open(csv_path, 'r') as file:
            reader = csv.DictReader(file)
            
            # Check if reader has fieldnames
            if not reader.fieldnames:
                app.logger.error("CSV file has no headers")
                return jsonify([])  # Return empty array if no headers
                
            # Print fieldnames for debugging
            app.logger.info(f"CSV fieldnames: {reader.fieldnames}")
            
            for row in reader:
                # Check for valid transaction ID
                if 'Transaction_ID' not in row or not row['Transaction_ID']:
                    app.logger.warning("Found row without Transaction_ID, skipping")
                    continue
                    
                # Clean up None values and ensure all keys are strings
                cleaned_row = {}
                for key, value in row.items():
                    if key is None:
                        continue  # Skip None keys
                    
                    # Ensure values are properly formatted
                    if value is None:
                        cleaned_row[str(key)] = ""
                    else:
                        cleaned_row[str(key)] = str(value)
                        
                txn_data.append(cleaned_row)
            
            app.logger.info(f"Loaded {len(txn_data)} transactions from CSV")
            
    except Exception as e:
        app.logger.error(f"Error loading transactions: {str(e)}")
        # Return empty list on error rather than failing
        return jsonify([])
    
    return jsonify(txn_data)

@app.route('/api/fraud_stats', methods=['GET'])
@admin_required
def api_fraud_stats():
    """API endpoint for fraud statistics"""
    try:
        with open('data/processed_transactions.csv', 'r') as file:
            reader = csv.DictReader(file)
            rows = list(reader)
            
            total = len(rows)
            fraud = sum(1 for r in rows if r['Is_Fraud'] == 'True')
            fraud_types = {}
            
            for row in rows:
                if row['Is_Fraud'] == 'True' and row['Fraud_Type']:
                    fraud_types[row['Fraud_Type']] = fraud_types.get(row['Fraud_Type'], 0) + 1
            
            return jsonify({
                'total_transactions': total,
                'fraud_count': fraud,
                'fraud_percentage': round(fraud/total*100, 2) if total > 0 else 0,
                'fraud_types': fraud_types
            })
    except FileNotFoundError:
        return jsonify({
            'total_transactions': 0,
            'fraud_count': 0,
            'fraud_percentage': 0,
            'fraud_types': {}
        })

@app.route('/api/feedback', methods=['POST'])
@admin_required
def transaction_feedback():
    """API endpoint to provide feedback on transaction classification
    
    This enables continuous learning in the fraud detection model by
    allowing admins to correct predictions and improve the model over time.
    """
    try:
        data = request.json
        app.logger.info(f"Received feedback data: {data}")
        
        if not data:
            app.logger.error("No JSON data received in feedback request")
            return jsonify({
                'status': 'error', 
                'message': 'No data provided'
            }), 400
            
        if 'transaction_id' not in data:
            app.logger.error("Missing transaction_id in feedback request")
            return jsonify({
                'status': 'error', 
                'message': 'Missing transaction_id field'
            }), 400
            
        if 'is_fraud' not in data:
            app.logger.error("Missing is_fraud in feedback request")
            return jsonify({
                'status': 'error', 
                'message': 'Missing is_fraud field'
            }), 400
        
        transaction_id = str(data['transaction_id'])
        is_fraud = bool(data['is_fraud'])
        feedback_source = data.get('source', 'admin')
        
        app.logger.info(f"Processing feedback for transaction {transaction_id}, is_fraud={is_fraud}, source={feedback_source}")
        
        # Process the feedback in the predictor
        # But don't fail if an error occurs
        try:
            predictor.feedback(transaction_id, is_fraud, feedback_source)
            app.logger.info(f"Predictor feedback processed for {transaction_id}")
        except Exception as e:
            app.logger.error(f"Error in predictor feedback: {str(e)}")
            # Continue anyway - we'll update the CSV at least
        
        # Update the transaction in the CSV
        update_transaction_csv(transaction_id, is_fraud)
        
        return jsonify({
            'status': 'success',
            'message': f'Feedback recorded for transaction {transaction_id}',
            'data': {
                'transaction_id': transaction_id,
                'is_fraud': is_fraud
            }
        })
    except Exception as e:
        app.logger.error(f"Error processing feedback: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Error processing feedback: {str(e)}'
        }), 500

def update_transaction_csv(transaction_id, is_fraud):
    """Update the transaction CSV file with new fraud status from feedback"""
    app.logger.info(f"Updating CSV for transaction {transaction_id}, is_fraud={is_fraud}")
    
    csv_path = 'data/processed_transactions.csv'
    if not os.path.exists(csv_path):
        app.logger.error(f"CSV file not found: {csv_path}")
        ensure_transaction_csv_exists()
    
    # Read the existing CSV file
    rows = []
    found = False
    
    try:
        with open(csv_path, 'r', newline='') as file:
            reader = csv.DictReader(file)
            
            if not reader.fieldnames:
                app.logger.error("CSV file has no headers")
                return
                
            fieldnames = reader.fieldnames
            
            for row in reader:
                if row.get('Transaction_ID') == transaction_id:
                    # Update the fraud status
                    found = True
                    app.logger.info(f"Found transaction {transaction_id} in CSV, updating fraud status")
                    row['Is_Fraud'] = str(is_fraud)
                    
                    if is_fraud:
                        # If marked as fraud but no fraud type, add generic one
                        if not row.get('Fraud_Type') or row.get('Fraud_Type') == '-':
                            row['Fraud_Type'] = 'Admin Flagged'
                    elif not is_fraud and row.get('Fraud_Type') != '-':
                        # If marked as legitimate, clear fraud type
                        row['Fraud_Type'] = '-'
                        
                rows.append(row)
        
        # If transaction not found, add a new entry
        if not found:
            app.logger.info(f"Transaction {transaction_id} not found in CSV, adding new entry")
            
            new_row = {field: '' for field in fieldnames}  # Initialize with empty values
            new_row['Transaction_ID'] = transaction_id
            new_row['Is_Fraud'] = str(is_fraud)
            new_row['Timestamp'] = datetime.now().isoformat()
            new_row['Amount'] = '0'  # Default amount
            new_row['Transaction_Type'] = 'Unknown'
            new_row['Fraud_Type'] = 'Admin Flagged' if is_fraud else '-'
            new_row['Fraud_Probability'] = '0.5'  # Default probability
            new_row['Status'] = 'COMPLETED'  # Mark as completed
            
            rows.append(new_row)
        
        # Write back to the CSV file
        with open(csv_path, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        
        app.logger.info(f"Successfully updated CSV for transaction {transaction_id}")
        
    except Exception as e:
        app.logger.error(f"Error updating transaction CSV: {str(e)}")
        raise

# Add model stats endpoint to monitor continuous learning
@app.route('/api/model_stats', methods=['GET'])
@admin_required
def model_stats():
    """API endpoint for model statistics and continuous learning info"""
    return jsonify({
        'model_info': {
            'last_retrained': predictor.last_retrain_time.isoformat(),
            'training_data_size': len(predictor.training_data),
            'min_samples_needed': predictor.min_samples_for_retraining,
            'next_retrain_due': (predictor.last_retrain_time + predictor.retrain_interval).isoformat(),
            'continuous_learning_enabled': predictor.continuous_learning_enabled
        },
        'fraud_types': predictor.get_fraud_type_summary(),
        'predictions_count': len(predictor.predictions_history)
    })

def log_transaction(transaction_data, result, status):
    """Log transaction to CSV file"""
    # Ensure CSV file exists
    ensure_transaction_csv_exists()
    
    # Append transaction to CSV
    with open('data/processed_transactions.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            transaction_data.get('Transaction_ID', 'Unknown'),
            transaction_data.get('Timestamp', datetime.now().isoformat()),
            transaction_data.get('Amount', 0),
            transaction_data.get('Sender_ID', 'Unknown'),
            transaction_data.get('Receiver_ID', 'Unknown'),
            transaction_data.get('Transaction_Type', 'Unknown'),
            transaction_data.get('Merchant_Type', 'Unknown'),
            transaction_data.get('Device_ID', 'Unknown'),
            transaction_data.get('Location', '0.0, 0.0'),
            result['is_fraud'],
            result.get('fraud_type', ''),
            result['fraud_probability'],
            status
        ])

# Error handlers
@app.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors"""
    return render_template('error.html', 
                          error_title="Page Not Found", 
                          error_message="The page you are looking for does not exist.",
                          error_code="404"), 404

@app.errorhandler(500)
def internal_server_error(e):
    """Handle 500 errors"""
    return render_template('error.html', 
                          error_title="Server Error", 
                          error_message="We're experiencing some technical difficulties.",
                          error_details="An unexpected error occurred on our servers.",
                          error_code="500"), 500

@app.errorhandler(403)
def forbidden(e):
    """Handle 403 errors"""
    return render_template('error.html', 
                          error_title="Access Denied", 
                          error_message="You do not have permission to access this resource.",
                          error_code="403"), 403

# Add admin login route
@app.route('/admin', methods=['GET', 'POST'])
def admin_login():
    """Admin login page for dashboard access"""
    if request.method == 'POST':
        # Simple password check - in a real app, use proper authentication
        if request.form.get('password') == ADMIN_PASSWORD:
            session['is_admin'] = True
            return redirect(url_for('dashboard'))
        else:
            return render_template('admin_login.html', error="Invalid password")
    return render_template('admin_login.html')

# Add admin logout route
@app.route('/admin/logout')
def admin_logout():
    """Admin logout"""
    session.pop('is_admin', None)
    return redirect(url_for('index'))

if __name__ == '__main__':
    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)
    
    # Ensure transaction CSV exists
    ensure_transaction_csv_exists()
    
    # Run the Flask app in debug mode
    app.run(debug=True) 