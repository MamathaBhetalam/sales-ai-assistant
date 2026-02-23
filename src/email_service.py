"""
Email Service: Compose and send analysis summary emails via SMTP.

Charts are embedded using CID (Content-ID) MIME attachments — the RFC-compliant
method that works in Gmail, Outlook, and all major email clients.
(data: URI images are blocked by most email clients for security reasons.)

Configure in .env:
  SMTP_USER      = your-email@gmail.com
  SMTP_PASSWORD  = your-app-password      (Gmail: use App Password, not account password)
  SMTP_HOST      = smtp.gmail.com         (optional, default)
  SMTP_PORT      = 587                    (optional, default)

Gmail setup: myaccount.google.com → Security → App Passwords
"""
import os
import base64
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from datetime import datetime


def _fig_to_png(fig) -> bytes | None:
    """Convert a Plotly figure to raw PNG bytes."""
    try:
        import plotly.io as pio
        return pio.to_image(fig, format="png", width=700, height=420, scale=1.5)
    except Exception:
        return None


def is_email_request(question: str) -> bool:
    """Detect if the user is asking to send an email summary."""
    q = question.lower()
    if any(phrase in q for phrase in [
        "send email", "email me", "mail me", "email this",
        "mail this", "email summary", "mail summary",
        "send summary", "email report", "mail report",
        "send report", "email the", "mail the",
        "send me a mail", "send me an email", "send me the",
        "to my email", "send me this", "send this to my",
        "send it to my", "my email", "my mail",
    ]):
        return True
    send_words  = ["send", "share", "forward", "deliver", "shoot"]
    email_words = ["email", "mail", "inbox"]
    return any(s in q for s in send_words) and any(e in q for e in email_words)


def compose_email(messages: list, role: str, kpis: dict,
                  figures: list = None) -> tuple[str, str, dict]:
    """
    Build email subject, HTML body, and inline image dict.

    Returns:
        subject       : str
        html_body     : str  — uses cid: references for charts
        inline_images : dict — {cid: png_bytes} for MIME attachment
    """
    date_str  = datetime.now().strftime("%B %d, %Y")
    time_str  = datetime.now().strftime("%H:%M")
    yr        = kpis.get("latest_year", "")
    rev       = kpis.get("revenue_latest", 0)
    prf       = kpis.get("profit_latest", 0)
    margin    = prf / rev * 100 if rev else 0
    yoy_rev   = kpis.get("yoy_revenue_growth", 0)
    yoy_color = "#16a34a" if yoy_rev >= 0 else "#dc2626"

    # ── KPI rows ───────────────────────────────────────────────────────────────
    kpi_rows = f"""
      <tr>
        <td style="padding:8px 14px;color:#64748b;font-size:14px">Total Revenue ({yr})</td>
        <td style="padding:8px 14px;text-align:right;font-weight:600;color:#2563eb;font-size:14px">${rev:,.0f}</td>
      </tr>
      <tr style="background:#f8fafc">
        <td style="padding:8px 14px;color:#64748b;font-size:14px">Total Profit ({yr})</td>
        <td style="padding:8px 14px;text-align:right;font-weight:600;color:#16a34a;font-size:14px">${prf:,.0f}</td>
      </tr>
      <tr>
        <td style="padding:8px 14px;color:#64748b;font-size:14px">Profit Margin</td>
        <td style="padding:8px 14px;text-align:right;font-weight:600;color:#d97706;font-size:14px">{margin:.1f}%</td>
      </tr>
      <tr style="background:#f8fafc">
        <td style="padding:8px 14px;color:#64748b;font-size:14px">YoY Revenue Growth</td>
        <td style="padding:8px 14px;text-align:right;font-weight:600;color:{yoy_color};font-size:14px">{yoy_rev:+.1f}%</td>
      </tr>
    """

    # ── Q&A turns ──────────────────────────────────────────────────────────────
    qa_html = ""
    for msg in messages:
        if msg["role"] == "user" and not is_email_request(msg["content"]):
            qa_html += f"""
      <div style="background:#f0f4fb;border-left:3px solid #2563eb;padding:10px 14px;
                  margin:10px 0;border-radius:4px">
        <span style="font-size:11px;color:#64748b;font-weight:700;
                     text-transform:uppercase;letter-spacing:0.05em">You asked</span><br>
        <span style="color:#1a2744;font-size:14px">{msg['content']}</span>
      </div>"""
        elif msg["role"] == "assistant" and not msg["content"].startswith("Sure! I'll prepare"):
            body = msg["content"].replace("\n", "<br>")
            qa_html += f"""
      <div style="background:#ffffff;border:1px solid #d1ddf0;padding:10px 14px;
                  margin:10px 0;border-radius:4px">
        <span style="font-size:11px;color:#2563eb;font-weight:700;
                     text-transform:uppercase;letter-spacing:0.05em">AI — {role} Mode</span><br>
        <span style="color:#1a2744;font-size:14px;line-height:1.65">{body}</span>
      </div>"""

    if not qa_html:
        qa_html = '<p style="color:#94a3b8;font-size:13px;font-style:italic">No conversation history included.</p>'

    # ── Charts: build CID references + collect raw bytes ──────────────────────
    inline_images = {}   # {cid: png_bytes}
    charts_section = ""

    if figures:
        chart_img_tags = []
        for i, fig in enumerate(figures):
            png_bytes = _fig_to_png(fig)
            if png_bytes:
                cid = f"chart_{i}"
                inline_images[cid] = png_bytes
                chart_img_tags.append(
                    f'<img src="cid:{cid}" width="100%"'
                    f' style="display:block;border-radius:8px;'
                    f'border:1px solid #e2e8f0;margin-bottom:12px">'
                )
        if chart_img_tags:
            charts_section = f"""
        <!-- Charts -->
        <tr>
          <td style="padding:0 32px 20px">
            <p style="margin:0 0 14px;font-size:12px;font-weight:700;color:#64748b;
                      text-transform:uppercase;letter-spacing:0.06em">
              Charts
            </p>
            {''.join(chart_img_tags)}
          </td>
        </tr>
        <!-- Divider -->
        <tr>
          <td style="padding:0 32px">
            <hr style="border:none;border-top:1px solid #e2e8f0;margin:0">
          </td>
        </tr>"""

    subject = f"Business AI Copilot — {role} Analysis Summary — {date_str}"

    html_body = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
</head>
<body style="margin:0;padding:0;background:#f0f4fb;
             font-family:Inter,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif">
  <table width="100%" cellpadding="0" cellspacing="0"
         style="background:#f0f4fb;padding:32px 16px">
    <tr><td align="center">
      <table width="600" cellpadding="0" cellspacing="0"
             style="background:#ffffff;border-radius:12px;
                    overflow:hidden;border:1px solid #d1ddf0;max-width:600px">

        <!-- Header -->
        <tr>
          <td style="background:linear-gradient(135deg,#2563eb 0%,#1e40af 100%);
                     padding:28px 32px">
            <p style="margin:0;font-size:22px;font-weight:700;color:#ffffff">
              Business AI Copilot
            </p>
            <p style="margin:8px 0 0;font-size:13px;color:#bfdbfe">
              {role} Mode Summary &nbsp;·&nbsp; {date_str} at {time_str}
            </p>
          </td>
        </tr>

        <!-- KPI Snapshot -->
        <tr>
          <td style="padding:28px 32px 20px">
            <p style="margin:0 0 14px;font-size:12px;font-weight:700;color:#64748b;
                      text-transform:uppercase;letter-spacing:0.06em">
              KPI Snapshot
            </p>
            <table width="100%" cellpadding="0" cellspacing="0"
                   style="border:1px solid #d1ddf0;border-radius:8px;
                          overflow:hidden;border-collapse:collapse">
              {kpi_rows}
            </table>
          </td>
        </tr>

        <!-- Divider -->
        <tr>
          <td style="padding:0 32px">
            <hr style="border:none;border-top:1px solid #e2e8f0;margin:0">
          </td>
        </tr>

        {charts_section}

        <!-- Conversation -->
        <tr>
          <td style="padding:20px 32px 28px">
            <p style="margin:0 0 14px;font-size:12px;font-weight:700;color:#64748b;
                      text-transform:uppercase;letter-spacing:0.06em">
              Analysis Conversation
            </p>
            {qa_html}
          </td>
        </tr>

        <!-- Footer -->
        <tr>
          <td style="background:#f8fafc;border-top:1px solid #e2e8f0;
                     padding:18px 32px;text-align:center">
            <p style="margin:0;font-size:12px;color:#94a3b8;line-height:1.6">
              Generated by <strong>Business AI Copilot</strong>
              &nbsp;·&nbsp; Powered by GPT-4o &amp; Streamlit<br>
              Superstore Dataset (2015–2018) &nbsp;·&nbsp; This is an automated summary.
            </p>
          </td>
        </tr>

      </table>
    </td></tr>
  </table>
</body>
</html>"""

    return subject, html_body, inline_images


def html_for_preview(html_body: str, inline_images: dict) -> str:
    """
    For the Streamlit preview iframe, swap cid: refs to data: URIs
    so charts are visible in the browser component.
    """
    preview = html_body
    for cid, png_bytes in inline_images.items():
        b64 = base64.b64encode(png_bytes).decode("utf-8")
        preview = preview.replace(
            f"cid:{cid}",
            f"data:image/png;base64,{b64}"
        )
    return preview


def send_email(to_address: str, subject: str, html_body: str,
               inline_images: dict = None) -> tuple[bool, str]:
    """
    Send email via SMTP using CID-embedded images (multipart/related).
    Returns (success: bool, message: str).
    """
    smtp_user = os.environ.get("SMTP_USER", "")
    smtp_pass = os.environ.get("SMTP_PASSWORD", "")
    smtp_host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
    smtp_port = int(os.environ.get("SMTP_PORT", "587"))

    if not smtp_user or not smtp_pass:
        return False, (
            "SMTP credentials not configured. "
            "Add SMTP_USER and SMTP_PASSWORD to your .env file."
        )

    try:
        # Root envelope
        msg_root = MIMEMultipart("mixed")
        msg_root["Subject"] = subject
        msg_root["From"]    = f"Business AI Copilot <{smtp_user}>"
        msg_root["To"]      = to_address

        if inline_images:
            # multipart/related wraps HTML + CID-attached images
            msg_related = MIMEMultipart("related")
            msg_related.attach(MIMEText(html_body, "html", "utf-8"))

            for cid, png_bytes in inline_images.items():
                img = MIMEImage(png_bytes, "png")
                img.add_header("Content-ID", f"<{cid}>")
                img.add_header("Content-Disposition", "inline", filename=f"{cid}.png")
                msg_related.attach(img)

            msg_root.attach(msg_related)
        else:
            msg_root.attach(MIMEText(html_body, "html", "utf-8"))

        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.ehlo()
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.sendmail(smtp_user, [to_address], msg_root.as_string())

        return True, f"Email sent successfully to {to_address}"

    except smtplib.SMTPAuthenticationError:
        return False, "Authentication failed. Use an App Password for Gmail (not your account password)."
    except smtplib.SMTPRecipientsRefused:
        return False, f"Invalid recipient address: {to_address}"
    except smtplib.SMTPException as e:
        return False, f"SMTP error: {str(e)}"
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"
