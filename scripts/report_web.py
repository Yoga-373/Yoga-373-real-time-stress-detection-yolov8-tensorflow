#this is report_web.py
import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"  # Disable Qt GUI support

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import logging
import json
import glob
import argparse
import threading
import cv2
import re
import base64 
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("stress_report.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ... rest of your code ...

# Asynchronous email sending
def send_email_async(subject, body, attachment_path=None):
    # Make sure to define or import your send_email function before using it
    thread = threading.Thread(target=send_email, args=(subject, body, attachment_path))
    thread.start()

class StressReportGenerator:
    def __init__(self, log_file, image_dir, output_dir):
        self.log_file = log_file
        self.stress_images_dir = image_dir
        self.report_output_dir = output_dir
        self.df = None
        self.image_files = []

    def parse_stress_events_json(self):
        """Extract stress_detected events from all rotated stress_events.json files."""
        base_path = self.log_file                 # e.g. '/home/yoga/.../logs/stress_events.json'
        log_dir  = os.path.dirname(base_path)
        base_fn  = os.path.basename(base_path)    # 'stress_events.json'
        pattern  = os.path.join(log_dir, base_fn + '*')
        
        logger.info(f"Looking for event logs: {pattern}")
        all_events = []
        
        for fp in sorted(glob.glob(pattern)):
            logger.info(f"Parsing {fp}")
            try:
                with open(fp, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        # extract the JSON payload
                        m = re.search(r'({.*})', line)
                        if not m:
                            logger.debug(f"Skipping non-JSON line: {line}")
                            continue
                        try:
                            evt = json.loads(m.group(1))
                            if evt.get('event_type') == 'stress_detected':
                                all_events.append(evt)
                        except json.JSONDecodeError as je:
                            logger.warning(f"Bad JSON in {fp}: {je}")
            except Exception as e:
                logger.error(f"Failed to read {fp}: {e}")
        
        if not all_events:
            logger.warning("No stress_detected events found across any logs.")
            self.df = pd.DataFrame(columns=['timestamp','stress_label','stress_percentage'])
            return True
        
        # build DataFrame
        df = pd.DataFrame({
            'timestamp':        [e['timestamp']         for e in all_events],
            'stress_label':     [e.get('stress_label')  for e in all_events],
            'stress_percentage':[e.get('stress_percentage',0) for e in all_events]
        })
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        self.df = df.sort_values('timestamp').reset_index(drop=True)
        logger.info(f"Parsed {len(self.df)} stress_detected events from {len(glob.glob(pattern))} files")
        return True


    def load_image_files(self):
        """Load stress image files for inclusion in the report"""
        logger.info(f"Looking for stress images in: {self.stress_images_dir}")
        try:
            # Get all jpg files
            self.image_files = glob.glob(os.path.join(self.stress_images_dir, "*.jpg"))
            logger.info(f"Found {len(self.image_files)} stress images")
        except Exception as e:
            logger.error(f"Error loading image files: {e}")
            return False
        return True

    def generate_daily_stress_chart(self):
        """Generate a chart showing stress events by day"""
        if self.df is None or len(self.df) == 0:
            logger.warning("No data available for daily stress chart")
            return None
        try:
            plt.figure(figsize=(12, 6))
            # Count stress events by date
            daily_counts = self.df.groupby(self.df['timestamp'].dt.date).size()
            ax = daily_counts.plot(kind='bar', color='skyblue')
            plt.title('Daily Stress Events', fontsize=16)
            plt.xlabel('Date', fontsize=12)
            plt.ylabel('Number of Stress Events', fontsize=12)
            plt.xticks(rotation=45)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            # Add count labels on bars
            for i, count in enumerate(daily_counts):
                ax.text(i, count + 0.1, str(count), ha='center')
            plt.tight_layout()
            daily_chart_path = os.path.join(self.report_output_dir, 'daily_stress_events.png')
            plt.savefig(daily_chart_path)
            plt.close()
            logger.info(f"Daily stress chart saved to: {daily_chart_path}")
            return daily_chart_path
        except Exception as e:
            logger.error(f"Error generating daily stress chart: {e}")
            return None

    def generate_hourly_stress_chart(self):
        """Generate a chart showing stress events by hour of day"""
        if self.df is None or len(self.df) == 0:
            logger.warning("No data available for hourly stress chart")
            return None
        try:
            plt.figure(figsize=(12, 6))
            hourly_counts = self.df.groupby(self.df['timestamp'].dt.hour).size()
            all_hours = pd.Series(0, index=range(24))
            hourly_counts = hourly_counts.add(all_hours, fill_value=0)
            ax = hourly_counts.plot(kind='bar', color='coral')
            plt.title('Hourly Distribution of Stress Events', fontsize=16)
            plt.xlabel('Hour of Day', fontsize=12)
            plt.ylabel('Number of Stress Events', fontsize=12)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            for i, count in enumerate(hourly_counts):
                if count > 0:
                    ax.text(i, count + 0.1, str(int(count)), ha='center')
            plt.tight_layout()
            hourly_chart_path = os.path.join(self.report_output_dir, 'hourly_stress_events.png')
            plt.savefig(hourly_chart_path)
            plt.close()
            logger.info(f"Hourly stress chart saved to: {hourly_chart_path}")
            return hourly_chart_path
        except Exception as e:
            logger.error(f"Error generating hourly stress chart: {e}")
            return None

    def generate_stress_level_distribution(self):
        """Generate a chart showing distribution of stress levels"""
        if self.df is None or len(self.df) == 0:
            logger.warning("No data available for stress level distribution")
            return None
        try:
            plt.figure(figsize=(10, 6))
            # Define bins and labels for stress percentages
            bins = [50, 70, 85, 100]
            labels = ['Mildly Stressed (51-70)', 'Stressed (71-85)', 'Highly Stressed (86-100)']
            # Filter for stress percentages >= 51 (as per classification)
            filtered_df = self.df[self.df['stress_percentage'] >= 51]
            stress_categories = pd.cut(filtered_df['stress_percentage'], bins=bins, labels=labels)
            stress_category_counts = stress_categories.value_counts().sort_index()
            colors = ['yellow', 'orange', 'red']
            ax = stress_category_counts.plot(kind='pie', autopct='%1.1f%%', 
                                             colors=colors, startangle=90,
                                             wedgeprops={'edgecolor': 'black', 'linewidth': 1})
            plt.title('Stress Level Distribution', fontsize=16)
            plt.ylabel('')  # Hide default label
            plt.axis('equal')
            levels_chart_path = os.path.join(self.report_output_dir, 'stress_level_distribution.png')
            plt.savefig(levels_chart_path)
            plt.close()
            logger.info(f"Stress level distribution saved to: {levels_chart_path}")
            return levels_chart_path
        except Exception as e:
            logger.error(f"Error generating stress level distribution: {e}")
            return None

    def create_report_with_examples(self, num_examples=3):
        """Create a visual report with stress examples"""
        if self.df is None or len(self.df) == 0 or len(self.image_files) == 0:
            logger.warning("Not enough data to create report with examples")
            return None
        try:
            num_examples = min(num_examples, len(self.image_files))
            plt.figure(figsize=(12, 12))
            gs = plt.GridSpec(3, 3)
            # Plot statistics charts
            ax1 = plt.subplot(gs[0, :])    # Daily trend
            ax2 = plt.subplot(gs[1, 0:2])    # Hourly distribution
            ax3 = plt.subplot(gs[1, 2])      # Stress level pie chart

            # Daily trend line chart
            daily_counts = self.df.groupby(self.df['timestamp'].dt.date).size()
            daily_counts.plot(kind='line', marker='o', ax=ax1)
            ax1.set_title('Daily Stress Trend')
            ax1.set_ylabel('Number of Events')
            ax1.grid(True, linestyle='--', alpha=0.7)
            ax1.tick_params(axis='x', rotation=45)

            # Hourly distribution bar chart
            hourly_counts = self.df.groupby(self.df['timestamp'].dt.hour).size()
            all_hours = pd.Series(0, index=range(24))
            hourly_counts = hourly_counts.add(all_hours, fill_value=0)
            hourly_counts.plot(kind='bar', ax=ax2, color='coral')
            ax2.set_title('Hourly Distribution')
            ax2.set_xlabel('Hour of Day')
            ax2.set_ylabel('Events')
            ax2.grid(axis='y', linestyle='--', alpha=0.7)

            # Stress level pie chart
            bins = [50, 70, 85, 100]
            labels = ['Mild', 'Medium', 'High']
            filtered_df = self.df[self.df['stress_percentage'] >= 51]
            stress_categories = pd.cut(filtered_df['stress_percentage'], bins=bins, labels=labels)
            stress_category_counts = stress_categories.value_counts().sort_index()
            colors = ['yellow', 'orange', 'red']
            stress_category_counts.plot(kind='pie', autopct='%1.1f%%', colors=colors, ax=ax3)
            ax3.set_title('Stress Levels')
            ax3.set_ylabel('')
            
            # Add example images
            for i in range(min(num_examples, 3)):
                img_path = self.image_files[i]
                img = cv2.imread(img_path)
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                filename = os.path.basename(img_path)
                match = re.search(r'stress_(.+)_(\d{8})_(\d{6})\.jpg', filename)
                if match:
                    stress_type = match.group(1).replace('_', ' ')
                    date_str = match.group(2)
                    time_str = match.group(3)
                    formatted_datetime = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]} {time_str[:2]}:{time_str[2:4]}:{time_str[4:]}"
                    img_title = f"{stress_type}\n{formatted_datetime}"
                else:
                    img_title = filename
                ax = plt.subplot(gs[2, i])
                ax.imshow(img)
                ax.set_title(img_title, fontsize=9)
                ax.axis('off')
            plt.tight_layout()
            combined_report_path = os.path.join(self.report_output_dir, f'stress_report_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
            plt.savefig(combined_report_path, dpi=150)
            plt.close()
            logger.info(f"Combined report created at: {combined_report_path}")
            return combined_report_path
        except Exception as e:
            logger.error(f"Error creating combined report: {e}")
            return None

    def generate_full_report(self, email_report=False):
        """Generate a complete stress report with all visualizations"""
        report_date = datetime.datetime.now().strftime("%Y-%m-%d")
        # Parse JSON file and load images
        if not self.parse_stress_events_json() or not self.load_image_files():
            logger.error("Failed to gather data for report")
            return False
        # Generate charts and combined report
        daily_chart = self.generate_daily_stress_chart()
        hourly_chart = self.generate_hourly_stress_chart()
        level_chart = self.generate_stress_level_distribution()
        combined_report = self.create_report_with_examples()
        # Inline charts as Base64
        with open(daily_chart,  "rb") as f:
            b64_daily  = base64.b64encode(f.read()).decode("utf-8")
        with open(hourly_chart, "rb") as f:
            b64_hourly = base64.b64encode(f.read()).decode("utf-8")
        with open(level_chart,  "rb") as f:
            b64_level  = base64.b64encode(f.read()).decode("utf-8")

        # Inline combined report image as Base64
        with open(combined_report, "rb") as img_f:
            b64_combined = base64.b64encode(img_f.read()).decode("utf-8")

        # Create HTML report
        html_report_path = os.path.join(self.report_output_dir, f'stress_report_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.html')
        try:
            if len(self.df) > 0:
                total_events = len(self.df)
                daily_counts = self.df.groupby(self.df['timestamp'].dt.date).size()
                most_stressful_day = daily_counts.idxmax()
                max_day_count = daily_counts.max()
                hourly_counts = self.df.groupby(self.df['timestamp'].dt.hour).size()
                most_stressful_hour = hourly_counts.idxmax()
                max_hour_count = hourly_counts.max()
                avg_stress = self.df['stress_percentage'].mean()
                first_event = self.df['timestamp'].min().strftime("%Y-%m-%d %H:%M:%S")
                last_event = self.df['timestamp'].max().strftime("%Y-%m-%d %H:%M:%S")
                stress_type_counts = self.df['stress_label'].value_counts()
                stress_types_html = ""
                for stress_type, count in stress_type_counts.items():
                    percent = (count / total_events) * 100
                    stress_types_html += f"<tr><td>{stress_type}</td><td>{count}</td><td>{percent:.1f}%</td></tr>"
            else:
                total_events = 0
                most_stressful_day = "N/A"
                max_day_count = 0
                most_stressful_hour = "N/A"
                max_hour_count = 0
                avg_stress = 0
                first_event = "N/A"
                last_event = "N/A"
                stress_types_html = "<tr><td colspan='3'>No data available</td></tr>"
            
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
            <title>Stress Detection Report - {report_date}</title>
            <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2 {{ color: #333366; }}
            .summary {{ background-color: #f5f5f5; padding: 15px; border-radius: 8px; }}
            .chart-container {{ margin: 20px 0; }}
            table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
            th, td {{ border: 1px solid #dddddd; text-align: left; padding: 8px; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .stress-examples {{ display: flex; flex-wrap: wrap; gap: 10px; }}
            .stress-example {{ max-width: 300px; }}
            .stress-example img {{ width: 100%; }}
            </style>
            </head>
            <body>
            <h1>Stress Detection Report</h1>
            <p>Report generated on: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            <div class="summary">
            <h2>Summary Statistics</h2>
            <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Total Stress Events Detected</td><td>{total_events}</td></tr>
            <tr><td>Most Stressful Day</td><td>{most_stressful_day} ({max_day_count} events)</td></tr>
            <tr><td>Most Stressful Hour</td><td>{most_stressful_hour}:00 ({max_hour_count} events)</td></tr>
            <tr><td>Average Stress Level</td><td>{avg_stress:.1f}%</td></tr>
            <tr><td>First Recorded Event</td><td>{first_event}</td></tr>
            <tr><td>Last Recorded Event</td><td>{last_event}</td></tr>
            </table>
            <h3>Stress Type Distribution</h3>
            <table>
            <tr><th>Stress Type</th><th>Count</th><th>Percentage</th></tr>
            {stress_types_html}
            </table>
            </div>
            

           <div class="chart-container">
           <h2>Daily Stress Trend</h2>
           <img src="data:image/png;base64,{b64_daily}" alt="Daily Stress Events" style="max-width: 100%;">
           </div>

            <div class="chart-container">
            <h2>Hourly Distribution</h2>
            <img src="data:image/png;base64,{b64_hourly}" alt="Hourly Stress Distribution" style="max-width: 100%;">
            </div>

            <div class="chart-container">
            <h2>Stress Level Distribution</h2>
            <img src="data:image/png;base64,{b64_level}" alt="Stress Level Distribution" style="max-width: 100%;">
            </div>



            </div>
            <h2>Recent Stress Examples</h2>
            <div class="stress-examples">
            """
            for i, img_path in enumerate(self.image_files[:5]):
                filename = os.path.basename(img_path)
                match = re.search(r'stress_(.+)_(\d{8})_(\d{6})\.jpg', filename)
                if match:
                    stress_type = match.group(1).replace('_', ' ')
                    date_str = match.group(2)
                    time_str = match.group(3)
                    formatted_datetime = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]} {time_str[:2]}:{time_str[2:4]}:{time_str[4:]}"
                    img_title = f"{stress_type} ({formatted_datetime})"
                else:
                    img_title = filename
                with open(img_path, "rb") as image_file:
                    encoded_img = base64.b64encode(image_file.read()).decode('utf-8')
                html_content += f"""
                    <div class="stress-example">
                    <h3>{img_title}</h3>
                    <img src="data:image/jpeg;base64,{encoded_img}" alt="Stress Example">
                    </div>
                    """


            html_content += """
            </div>
            </body>
            </html>
            """
            with open(html_report_path, 'w') as f:
                f.write(html_content)
            logger.info(f"HTML report generated: {html_report_path}")
            # Email report if requested
            if email_report and combined_report:
                email_subject = f"Stress Detection Report - {report_date}"
                email_body = f"""
                Stress Detection Report for {report_date}
                Total Events: {total_events}
                Most Stressful Day: {most_stressful_day} ({max_day_count} events)
                Most Stressful Hour: {most_stressful_hour}:00 ({max_hour_count} events)
                Average Stress Level: {avg_stress:.1f}%
                Please see the attached report and the full HTML report for more details.
                """
                send_email_async(email_subject, email_body, attachment_path=combined_report)
                logger.info(f"Report email sent with attachment: {combined_report}")
            return html_report_path
        except Exception as e:
            logger.error(f"Error generating HTML report: {e}")
            return None

def main():
    parser = argparse.ArgumentParser(description='Generate Stress Detection Reports')
    parser.add_argument('--log', type=str, default='/home/yoga/STRESS_DETECTION/STRESSED/logs/stress_events.json', 
                        help='Path to the stress events JSON file')
    parser.add_argument('--images', type=str, default='/home/yoga/STRESS_DETECTION/STRESSED/captured_stressed/',
                        help='Directory containing stress images')
    parser.add_argument('--output', type=str, default='/home/yoga/STRESS_DETECTION/STRESSED/reports/',
                        help='Directory to store generated reports')
    parser.add_argument('--email', action='store_true', 
                        help='Email the report when complete')
    args = parser.parse_args()

    report_generator = StressReportGenerator(
        log_file=args.log,
        image_dir=args.images,
        output_dir=args.output
    )
    report_path = report_generator.generate_full_report(email_report=args.email)
    if report_path:
        print(f"✅ Report successfully generated: {report_path}")
        return 0
    else:
        print("❌ Failed to generate report")
        return 1

if __name__ == "__main__":
    main()
