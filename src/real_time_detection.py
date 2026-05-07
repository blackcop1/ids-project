"""Real-time network packet sniffer and intrusion detection"""

import numpy as np
from scapy.all import sniff, IP, TCP, UDP, ICMP
from src.utils import Logger, print_section
import time
from datetime import datetime


class PacketSniffer:
    """Capture and analyze network packets"""
    
    def __init__(self, interface=None):
        """Initialize packet sniffer
        
        Args:
            interface: Network interface to sniff on (e.g., 'eth0', 'wlan0')
        """
        self.interface = interface
        self.logger = Logger()
        self.packets = []
    
    def extract_packet_features(self, packet):
        """Extract relevant features from a packet
        
        Args:
            packet: Scapy packet object
        
        Returns:
            Dictionary of extracted features
        """
        features = {}
        
        try:
            if IP in packet:
                features['src_ip'] = packet[IP].src
                features['dst_ip'] = packet[IP].dst
                features['ttl'] = packet[IP].ttl
                features['packet_length'] = len(packet)
                features['protocol'] = packet[IP].proto
                
                if TCP in packet:
                    features['src_port'] = packet[TCP].sport
                    features['dst_port'] = packet[TCP].dport
                    features['tcp_flags'] = packet[TCP].flags
                elif UDP in packet:
                    features['src_port'] = packet[UDP].sport
                    features['dst_port'] = packet[UDP].dport
                elif ICMP in packet:
                    features['icmp_type'] = packet[ICMP].type
        
        except Exception as e:
            self.logger.warning(f'Error extracting features: {str(e)}')
        
        return features
    
    def packet_callback(self, packet):
        """Callback function for each sniffed packet"""
        features = self.extract_packet_features(packet)
        if features:
            self.packets.append(features)
    
    def start_sniffing(self, packet_count=100):
        """Start sniffing packets
        
        Args:
            packet_count: Number of packets to capture
        """
        print_section('Starting Packet Sniffing')
        
        self.logger.info(f'Sniffing {packet_count} packets on {self.interface}...')
        
        try:
            sniff(
                prn=self.packet_callback,
                iface=self.interface,
                count=packet_count,
                store=False,
                verbose=True
            )
            self.logger.info(f'Captured {len(self.packets)} packets')
        except PermissionError:
            self.logger.error('Permission denied. Run with sudo/admin privileges')
        except Exception as e:
            self.logger.error(f'Error during sniffing: {str(e)}')


class RealtimeDetector:
    """Real-time intrusion detection using trained models"""
    
    def __init__(self, model, scaler, label_encoder, feature_columns):
        """Initialize real-time detector
        
        Args:
            model: Trained ML model
            scaler: Feature scaler
            label_encoder: Label encoder
            feature_columns: List of expected feature column names
        """
        self.model = model
        self.scaler = scaler
        self.label_encoder = label_encoder
        self.feature_columns = feature_columns
        self.logger = Logger()
        self.alert_threshold = 0.7
        self.anomalies_detected = 0
        self.packets_analyzed = 0
    
    def extract_features_from_packet(self, packet_dict):
        """Extract ML features from packet dictionary
        
        Args:
            packet_dict: Dictionary of packet features
        
        Returns:
            Feature vector as numpy array
        """
        # This is a simplified example
        # In production, you'd extract more sophisticated features
        features = []
        for col in self.feature_columns:
            if col in packet_dict:
                features.append(float(packet_dict[col]))
            else:
                features.append(0.0)
        return np.array(features).reshape(1, -1)
    
    def predict_packet(self, packet_dict):
        """Predict if packet is anomalous
        
        Args:
            packet_dict: Dictionary of packet features
        
        Returns:
            Tuple of (prediction, confidence, label)
        """
        try:
            # Extract features
            features = self.extract_features_from_packet(packet_dict)
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Make prediction
            if hasattr(self.model, 'predict_proba'):
                prediction_proba = self.model.predict_proba(features_scaled)[0]
                prediction = np.argmax(prediction_proba)
                confidence = np.max(prediction_proba)
            else:
                # For neural networks
                prediction_proba = self.model.predict(features_scaled, verbose=0)[0]
                prediction = np.argmax(prediction_proba)
                confidence = np.max(prediction_proba)
            
            # Decode label
            label = self.label_encoder.inverse_transform([prediction])[0]
            
            self.packets_analyzed += 1
            
            # Check if anomalous
            is_anomalous = (prediction != 0) and (confidence > self.alert_threshold)
            
            if is_anomalous:
                self.anomalies_detected += 1
            
            return prediction, confidence, label, is_anomalous
        
        except Exception as e:
            self.logger.error(f'Error in prediction: {str(e)}')
            return None, None, None, False
    
    def generate_alert(self, packet_dict, prediction, confidence, label):
        """Generate alert for detected anomaly
        
        Args:
            packet_dict: Packet features
            prediction: Model prediction
            confidence: Prediction confidence
            label: Attack type label
        """
        alert_msg = f"""
╔════════════════════════════════════════════════════════════╗
║                    🚨 ANOMALY DETECTED 🚨                  ║
╠════════════════════════════════════════════════════════════╣
║ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
║ Source IP: {packet_dict.get('src_ip', 'Unknown')}
║ Destination IP: {packet_dict.get('dst_ip', 'Unknown')}
║ Source Port: {packet_dict.get('src_port', 'Unknown')}
║ Destination Port: {packet_dict.get('dst_port', 'Unknown')}
║ Attack Type: {label}
║ Confidence: {confidence:.2%}
║ Protocol: {packet_dict.get('protocol', 'Unknown')}
║ Packet Size: {packet_dict.get('packet_length', 'Unknown')} bytes
╚════════════════════════════════════════════════════════════╝
        """
        
        self.logger.warning(alert_msg)
        
        # Log to file
        with open('logs/ids_alerts.log', 'a') as f:
            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - "
                   f"Anomaly: {label} from {packet_dict.get('src_ip', 'Unknown')} "
                   f"to {packet_dict.get('dst_ip', 'Unknown')} "
                   f"(Confidence: {confidence:.2%})\n")
    
    def process_packet(self, packet_dict):
        """Process a single packet and generate alerts if needed
        
        Args:
            packet_dict: Dictionary of packet features
        """
        prediction, confidence, label, is_anomalous = self.predict_packet(packet_dict)
        
        if is_anomalous:
            self.generate_alert(packet_dict, prediction, confidence, label)
    
    def get_statistics(self):
        """Get detection statistics
        
        Returns:
            Dictionary with statistics
        """
        stats = {
            'packets_analyzed': self.packets_analyzed,
            'anomalies_detected': self.anomalies_detected,
            'detection_rate': (self.anomalies_detected / max(self.packets_analyzed, 1)) * 100
        }
        return stats
    
    def print_statistics(self):
        """Print detection statistics"""
        stats = self.get_statistics()
        print_section('Detection Statistics')
        self.logger.info(f'Total packets analyzed: {stats["packets_analyzed"]}')
        self.logger.info(f'Anomalies detected: {stats["anomalies_detected"]}')
        self.logger.info(f'Detection rate: {stats["detection_rate"]:.2f}%')
