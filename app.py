"""
Streamlit UI for the telecom RAG chatbot.

Run with:
	streamlit run app.py
"""

import os

os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from dotenv import load_dotenv
import streamlit as st

from rag_chain import build_chain


load_dotenv()


SAMPLE_QUESTIONS = [
	"Why is my mobile internet so slow?",
	"My calls keep dropping. What should I do?",
	"How do I activate international roaming?",
	"Why is my bill higher than usual this month?",
	"My phone says SIM not detected after restart.",
	"How do I enable Wi-Fi calling?",
]


st.set_page_config(
	page_title="Telecom Care Assistant",
	page_icon="Signal",
	layout="wide",
	initial_sidebar_state="expanded",
)


st.markdown(
	"""
<style>
	:root {
		--bg-0: #07111f;
		--bg-1: #0c1728;
		--panel: rgba(14, 28, 48, 0.82);
		--panel-strong: rgba(10, 20, 36, 0.96);
		--stroke: rgba(121, 147, 182, 0.22);
		--text: #e8eef7;
		--muted: #8ea4bf;
		--accent: #36c2d6;
		--accent-2: #7ef0c0;
		--warning: #ffd36e;
	}

	.stApp {
		background:
			radial-gradient(circle at top left, rgba(54, 194, 214, 0.18), transparent 30%),
			radial-gradient(circle at top right, rgba(126, 240, 192, 0.12), transparent 26%),
			linear-gradient(160deg, var(--bg-0), var(--bg-1));
		color: var(--text);
	}

	.block-container {
		padding-top: 1.2rem;
		padding-bottom: 2rem;
		max-width: 1240px;
	}

	.hero-card {
		background: linear-gradient(135deg, rgba(18, 35, 58, 0.95), rgba(9, 19, 34, 0.95));
		border: 1px solid var(--stroke);
		border-radius: 24px;
		padding: 1.4rem 1.5rem;
		box-shadow: 0 20px 60px rgba(0, 0, 0, 0.22);
	}

	.hero-kicker {
		color: var(--accent-2);
		letter-spacing: 0.18em;
		text-transform: uppercase;
		font-size: 0.74rem;
		font-weight: 700;
		margin-bottom: 0.45rem;
	}

	.hero-title {
		color: var(--text);
		font-size: clamp(2rem, 4vw, 3.25rem);
		line-height: 1.03;
		font-weight: 800;
		margin-bottom: 0.65rem;
	}

	.hero-copy {
		color: var(--muted);
		font-size: 1rem;
		line-height: 1.6;
		max-width: 70ch;
	}

	.info-strip {
		display: grid;
		grid-template-columns: repeat(3, minmax(0, 1fr));
		gap: 0.75rem;
		margin-top: 1rem;
	}

	.info-tile {
		background: rgba(7, 15, 28, 0.5);
		border: 1px solid var(--stroke);
		border-radius: 18px;
		padding: 0.85rem 1rem;
	}

	.info-label {
		color: var(--muted);
		font-size: 0.78rem;
		text-transform: uppercase;
		letter-spacing: 0.12em;
		margin-bottom: 0.25rem;
	}

	.info-value {
		color: var(--text);
		font-size: 1rem;
		font-weight: 700;
	}

	.chat-shell {
		background: var(--panel);
		border: 1px solid var(--stroke);
		border-radius: 24px;
		padding: 1rem;
		box-shadow: 0 20px 50px rgba(0, 0, 0, 0.18);
	}

	.sidebar-card {
		background: var(--panel-strong);
		border: 1px solid var(--stroke);
		border-radius: 20px;
		padding: 1rem;
	}

	.sample-note {
		color: var(--muted);
		font-size: 0.88rem;
		line-height: 1.5;
		margin-top: 0.35rem;
		margin-bottom: 0.75rem;
	}

	.stChatMessage {
		border-radius: 18px;
	}

	.stChatMessage[data-testid="stChatMessage"] {
		background: rgba(7, 16, 30, 0.4);
		border: 1px solid rgba(121, 147, 182, 0.15);
	}

	.stButton > button {
		border-radius: 999px;
		border: 1px solid rgba(126, 240, 192, 0.22);
		background: linear-gradient(135deg, rgba(54, 194, 214, 0.18), rgba(126, 240, 192, 0.12));
		color: var(--text);
		font-weight: 600;
	}

	.stButton > button:hover {
		border-color: rgba(126, 240, 192, 0.4);
		transform: translateY(-1px);
	}

	div[data-testid="stChatInput"] {
		background: rgba(8, 17, 31, 0.92);
		border: 1px solid var(--stroke);
		border-radius: 18px;
	}

	@media (max-width: 900px) {
		.info-strip {
			grid-template-columns: 1fr;
		}
	}
</style>
""",
	unsafe_allow_html=True,
)


@st.cache_resource
def get_chain():
	return build_chain()


def initialize_state() -> None:
	if "messages" not in st.session_state:
		st.session_state.messages = [
			{
				"role": "assistant",
				"content": (
					"I can help with mobile service issues, billing questions, SIM problems, "
					"roaming, and common troubleshooting steps. Ask a question or choose a sample below."
				),
			}
		]

	if "pending_question" not in st.session_state:
		st.session_state.pending_question = None


def render_header() -> None:
	st.markdown(
		"""
		<div class="hero-card">
			<div class="hero-kicker">Telecom support assistant</div>
			<div class="hero-title">Resolve customer issues with a focused RAG chatbot</div>
			<div class="hero-copy">
				This interface is tuned for fast customer care: ask about connectivity, SIM problems,
				roaming, billing, device setup, or app support. The assistant replies from the telecom
				knowledge base and past resolved cases.
			</div>
			<div class="info-strip">
				<div class="info-tile">
					<div class="info-label">Mode</div>
					<div class="info-value">Retrieval-Augmented Chat</div>
				</div>
				<div class="info-tile">
					<div class="info-label">Coverage</div>
					<div class="info-value">FAQ + Support Tickets</div>
				</div>
				<div class="info-tile">
					<div class="info-label">Fallback</div>
					<div class="info-value">Call 611 or use the app</div>
				</div>
			</div>
		</div>
		""",
		unsafe_allow_html=True,
	)


def render_sidebar() -> None:
	with st.sidebar:
		st.markdown(
			"""
			<div class="sidebar-card">
				<div class="hero-kicker">Conversation tools</div>
				<div style="font-size: 1.2rem; font-weight: 800; color: var(--text); margin-bottom: 0.35rem;">
					Quick actions
				</div>
				<div class="sample-note">
					Pick a common question, or clear the transcript to start fresh.
				</div>
			</div>
			""",
			unsafe_allow_html=True,
		)

		st.markdown("<div style='height: 0.7rem;'></div>", unsafe_allow_html=True)

		for question in SAMPLE_QUESTIONS:
			if st.button(question, use_container_width=True, key=f"sample_{question}"):
				st.session_state.pending_question = question

		st.markdown("<div style='height: 0.35rem;'></div>", unsafe_allow_html=True)

		if st.button("Clear conversation", use_container_width=True):
			st.session_state.messages = [
				{
					"role": "assistant",
					"content": (
						"I can help with mobile service issues, billing questions, SIM problems, "
						"roaming, and common troubleshooting steps. Ask a question or choose a sample below."
					),
				}
			]
			st.session_state.pending_question = None
			st.rerun()

		st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
		st.caption("The assistant uses only the retrieved telecom context when answering.")


def render_messages() -> None:
	for message in st.session_state.messages:
		with st.chat_message(message["role"]):
			st.markdown(message["content"])


def stream_answer(question: str):
	chain = get_chain()

	with st.spinner("Searching telecom knowledge base..."):
		for chunk in chain.stream(question):
			text = getattr(chunk, "content", str(chunk))
			if text:
				yield text


def main() -> None:
	initialize_state()
	render_sidebar()
	render_header()

	st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
	st.markdown("<div class='chat-shell'>", unsafe_allow_html=True)
	render_messages()

	question = st.chat_input("Ask about billing, roaming, SIM, internet, or call issues...")
	if st.session_state.pending_question:
		question = st.session_state.pending_question
		st.session_state.pending_question = None

	if question:
		st.session_state.messages.append({"role": "user", "content": question})
		with st.chat_message("user"):
			st.markdown(question)

		with st.chat_message("assistant"):
			try:
				answer = st.write_stream(stream_answer(question))
			except Exception as exc:
				answer = (
					"I could not generate a response because the chain failed to run. "
					f"Error: {exc}"
				)
				st.error(answer)

		st.session_state.messages.append({"role": "assistant", "content": answer})

	st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
	main()
