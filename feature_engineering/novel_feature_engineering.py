import pandas as pd
import plotly.express as px

# Load the data
df = pd.read_csv('ALLFLOWMETER_HIKARI2021.csv')

# Calculate temporal features
df['fwd_iat_diff'] = df['fwd_iat.max'] - df['fwd_iat.min']
df['bwd_iat_diff'] = df['bwd_iat.max'] - df['bwd_iat.min']
df['flow_iat_diff'] = df['flow_iat.max'] - df['flow_iat.min']

# Normalized packet counts and bytes
df['fwd_pkts_per_duration'] = df['fwd_pkts_tot'] / (df['flow_duration'] + 1e-5)
df['bwd_pkts_per_duration'] = df['bwd_pkts_tot'] / (df['flow_duration'] + 1e-5)
df['fwd_bytes_per_duration'] = df['fwd_subflow_bytes'] / (df['flow_duration'] + 1e-5)
df['bwd_bytes_per_duration'] = df['bwd_subflow_bytes'] / (df['flow_duration'] + 1e-5)

# Statistical features for packet sizes and payloads
df['fwd_payload_range'] = df['fwd_pkts_payload.max'] - df['fwd_pkts_payload.min']
df['bwd_payload_range'] = df['bwd_pkts_payload.max'] - df['bwd_pkts_payload.min']
df['fwd_payload_variance'] = df['fwd_pkts_payload.std']**2
df['bwd_payload_variance'] = df['bwd_pkts_payload.std']**2
df['flow_payload_range'] = df['flow_pkts_payload.max'] - df['flow_pkts_payload.min']
df['flow_payload_variance'] = df['flow_pkts_payload.std']**2

# Bulk data features
df['total_bulk_bytes'] = df['fwd_bulk_bytes'] + df['bwd_bulk_bytes']
df['total_bulk_packets'] = df['fwd_bulk_packets'] + df['bwd_bulk_packets']
df['fwd_bulk_data_rate'] = df['fwd_bulk_bytes'] / (df['flow_duration'] + 1e-5)
df['bwd_bulk_data_rate'] = df['bwd_bulk_bytes'] / (df['flow_duration'] + 1e-5)

# Packet Header features
df['fwd_header_size_ratio'] = df['fwd_header_size_tot'] / (df['fwd_pkts_tot'] + 1e-5)
df['bwd_header_size_ratio'] = df['bwd_header_size_tot'] / (df['bwd_pkts_tot'] + 1e-5)

# Activity and Idle Time Features
df['active_time_range'] = df['active.max'] - df['active.min']
df['idle_time_range'] = df['idle.max'] - df['idle.min']
df['active_time_ratio'] = df['active.tot'] / (df['active.avg'] + 1e-5)
df['idle_time_ratio'] = df['idle.tot'] / (df['idle.avg'] + 1e-5)

# Flags combinations and interaction terms
df['fin_syn_flags_sum'] = df['flow_FIN_flag_count'] + df['flow_SYN_flag_count']
df['rst_ack_flags_sum'] = df['flow_RST_flag_count'] + df['flow_ACK_flag_count']

# Packet Inter-Arrival Times (IAT) ratios
df['iat_fwd_bwd_ratio'] = df['fwd_iat.avg'] / (df['bwd_iat.avg'] + 1e-5)
df['iat_total_avg'] = (df['fwd_iat.avg'] + df['bwd_iat.avg']) / 2

# Save to new CSV
df.to_csv('processed_flow_data.csv', index=False)
