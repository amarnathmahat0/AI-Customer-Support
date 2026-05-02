"""
order_db.py — SQLite mock order database with 20 sample orders.

Provides functions to initialize the database and query order status
for the MCP get_order_status tool.
"""

import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Optional
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# 20 sample orders covering all statuses
SAMPLE_ORDERS = [
    ("ORD-100001", "user_alice", "Wireless Headphones", "DELIVERED",
     "2024-01-10", "2024-01-15", "2024-01-14", 89.99, "John Smith", "123 Main St, New York, NY 10001"),
    ("ORD-100002", "user_bob", "Mechanical Keyboard", "SHIPPED",
     "2024-01-18", "2024-01-25", None, 149.99, "Bob Johnson", "456 Oak Ave, Chicago, IL 60601"),
    ("ORD-100003", "user_carol", "USB-C Hub", "PROCESSING",
     "2024-01-20", "2024-01-27", None, 45.00, "Carol Williams", "789 Pine Rd, Austin, TX 78701"),
    ("ORD-100004", "user_dave", "Laptop Stand", "DELIVERED",
     "2024-01-05", "2024-01-10", "2024-01-09", 55.50, "Dave Brown", "321 Elm St, Seattle, WA 98101"),
    ("ORD-100005", "user_alice", "Webcam 4K", "CANCELLED",
     "2024-01-12", "2024-01-19", None, 120.00, "John Smith", "123 Main St, New York, NY 10001"),
    ("ORD-100006", "user_eve", "Ergonomic Mouse", "SHIPPED",
     "2024-01-19", "2024-01-26", None, 75.99, "Eve Davis", "654 Maple Dr, Boston, MA 02101"),
    ("ORD-100007", "user_frank", "Monitor 27inch", "DELIVERED",
     "2023-12-28", "2024-01-05", "2024-01-04", 399.99, "Frank Miller", "987 Cedar Ln, Denver, CO 80201"),
    ("ORD-100008", "user_grace", "Desk Lamp LED", "PROCESSING",
     "2024-01-21", "2024-01-28", None, 35.99, "Grace Wilson", "147 Birch Blvd, Miami, FL 33101"),
    ("ORD-100009", "user_henry", "Noise Cancelling Earbuds", "REFUNDED",
     "2024-01-08", "2024-01-15", "2024-01-14", 199.99, "Henry Taylor", "258 Willow Way, Portland, OR 97201"),
    ("ORD-100010", "user_iris", "Phone Stand", "DELIVERED",
     "2024-01-02", "2024-01-08", "2024-01-07", 22.50, "Iris Anderson", "369 Spruce St, Nashville, TN 37201"),
    ("ORD-100011", "user_jack", "Smart Watch", "SHIPPED",
     "2024-01-20", "2024-01-28", None, 299.99, "Jack Thomas", "741 Aspen Ave, Phoenix, AZ 85001"),
    ("ORD-100012", "user_kate", "Cable Management Kit", "DELIVERED",
     "2024-01-14", "2024-01-20", "2024-01-19", 18.99, "Kate Jackson", "852 Poplar Pl, Atlanta, GA 30301"),
    ("ORD-100013", "user_leo", "Gaming Controller", "PROCESSING",
     "2024-01-22", "2024-01-29", None, 65.00, "Leo White", "963 Oak Dr, Minneapolis, MN 55401"),
    ("ORD-100014", "user_mia", "Portable Charger", "SHIPPED",
     "2024-01-17", "2024-01-24", None, 49.99, "Mia Harris", "174 Pine Ave, San Diego, CA 92101"),
    ("ORD-100015", "user_noah", "Microphone USB", "DELIVERED",
     "2024-01-09", "2024-01-16", "2024-01-15", 85.00, "Noah Martin", "285 Elm Blvd, Detroit, MI 48201"),
    ("ORD-100016", "user_olivia", "Streaming Deck", "DELAYED",
     "2024-01-16", "2024-01-23", None, 149.99, "Olivia Garcia", "396 Maple St, Las Vegas, NV 89101"),
    ("ORD-100017", "user_peter", "Dual Monitor Arm", "SHIPPED",
     "2024-01-19", "2024-01-26", None, 89.00, "Peter Lee", "507 Cedar Ave, Charlotte, NC 28201"),
    ("ORD-100018", "user_quinn", "Keyboard Wrist Rest", "DELIVERED",
     "2024-01-11", "2024-01-17", "2024-01-16", 28.99, "Quinn Robinson", "618 Birch Ln, Columbus, OH 43201"),
    ("ORD-100019", "user_rose", "Blue Light Glasses", "CANCELLED",
     "2024-01-13", "2024-01-20", None, 39.99, "Rose Walker", "729 Willow Dr, Indianapolis, IN 46201"),
    ("ORD-100020", "user_sam", "Thunderbolt Dock", "PROCESSING",
     "2024-01-23", "2024-01-30", None, 189.99, "Sam Young", "840 Spruce Blvd, Sacramento, CA 95801"),
]


def get_db_path() -> str:
    """Return the path to the orders SQLite database."""
    return os.getenv("ORDER_DB", "./orders.db")


def init_order_db() -> None:
    """
    Initialize the orders SQLite database with the schema and sample data.
    
    Creates the orders table if it doesn't exist and populates it with
    20 sample orders if the table is empty.
    """
    db_path = get_db_path()
    logger.info(f"Initializing order database at {db_path}")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS orders (
            order_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            product_name TEXT NOT NULL,
            status TEXT NOT NULL,
            order_date TEXT NOT NULL,
            estimated_delivery TEXT,
            actual_delivery TEXT,
            total_amount REAL NOT NULL,
            customer_name TEXT NOT NULL,
            shipping_address TEXT NOT NULL,
            tracking_number TEXT,
            carrier TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Only insert if empty
    cursor.execute("SELECT COUNT(*) FROM orders")
    count = cursor.fetchone()[0]
    
    if count == 0:
        carriers = ["UPS", "FedEx", "USPS", "DHL"]
        for i, order in enumerate(SAMPLE_ORDERS):
            tracking = f"1Z{''.join([str((i*7+j)%10) for j in range(16)])}"
            carrier = carriers[i % len(carriers)]
            cursor.execute("""
                INSERT INTO orders 
                (order_id, user_id, product_name, status, order_date, 
                 estimated_delivery, actual_delivery, total_amount, 
                 customer_name, shipping_address, tracking_number, carrier)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (*order, tracking, carrier))
        logger.info(f"Inserted {len(SAMPLE_ORDERS)} sample orders")
    
    conn.commit()
    conn.close()
    logger.info("Order database initialized successfully")


def get_order_status(order_id: str) -> dict:
    """
    Query order status from the SQLite database by order_id.
    
    Args:
        order_id: The order identifier in format ORD-XXXXXX
        
    Returns:
        Dictionary with order details or error message if not found.
    """
    db_path = get_db_path()
    
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT order_id, user_id, product_name, status, order_date,
                   estimated_delivery, actual_delivery, total_amount,
                   customer_name, shipping_address, tracking_number, carrier
            FROM orders
            WHERE order_id = ?
        """, (order_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return {
                "found": False,
                "error": f"Order {order_id} not found in our system.",
                "order_id": order_id
            }
        
        order = dict(row)
        order["found"] = True
        
        # Add human-readable status message
        status_messages = {
            "PROCESSING": "Your order is being processed and will ship soon.",
            "SHIPPED": f"Your order is on its way via {order['carrier']}! Tracking: {order['tracking_number']}",
            "DELIVERED": f"Your order was delivered on {order['actual_delivery']}.",
            "CANCELLED": "This order has been cancelled. If you did not request this, please contact support.",
            "REFUNDED": "A refund has been issued for this order.",
            "DELAYED": "We apologize — your order has been delayed. Our team is working to resolve this."
        }
        order["status_message"] = status_messages.get(order["status"], "Status unknown.")
        
        return order
        
    except Exception as e:
        logger.error(f"Error querying order {order_id}: {e}")
        return {
            "found": False,
            "error": f"Database error while looking up order {order_id}.",
            "order_id": order_id
        }


def list_orders_for_user(user_id: str) -> list:
    """
    List all orders for a given user_id.
    
    Args:
        user_id: The user identifier
        
    Returns:
        List of order dictionaries for the user.
    """
    db_path = get_db_path()
    
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT order_id, product_name, status, order_date, total_amount
            FROM orders
            WHERE user_id = ?
            ORDER BY order_date DESC
        """, (user_id,))
        
        rows = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return rows
        
    except Exception as e:
        logger.error(f"Error listing orders for user {user_id}: {e}")
        return []


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    init_order_db()
    print("Database initialized. Testing order lookup...")
    result = get_order_status("ORD-100001")
    print(f"ORD-100001: {result}")
    result = get_order_status("ORD-999999")
    print(f"ORD-999999 (not found): {result}")
