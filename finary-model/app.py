import gradio as gr

from api_service import (
    ClassifyRequest,
    PredictRequest,
    SideHustleRequest,
    classify_financial_scenario,
    predict_insight,
    recommend_side_hustle,
)


def run_classification(
    monthly_income: float,
    monthly_expense_total: float,
    actual_savings: float,
    budget_goal: float,
    emergency_fund: float,
):
    try:
        payload = ClassifyRequest(
            monthly_income=monthly_income,
            monthly_expense_total=monthly_expense_total,
            actual_savings=actual_savings,
            budget_goal=budget_goal,
            emergency_fund=emergency_fund,
        )
        response = classify_financial_scenario(payload)
        return response.model_dump()
    except Exception as exc:
        return {"error": str(exc)}


def run_insight(
    income: float,
    expense: float,
    savings: float,
    target_tabungan: float,
    loan_payment: float,
    emergency_fund: float,
):
    try:
        payload = PredictRequest(
            income=income,
            expense=expense,
            savings=savings,
            target_tabungan=target_tabungan,
            loan_payment=loan_payment,
            emergency_fund=emergency_fund,
        )
        response = predict_insight(payload)
        return response.model_dump()
    except Exception as exc:
        return {"error": str(exc)}


def run_side_hustle(
    experience_level: str,
    available_hours_per_week: int,
    interest_category: str,
):
    try:
        payload = SideHustleRequest(
            experience_level=experience_level,
            available_hours_per_week=available_hours_per_week,
            interest_category=interest_category,
        )
        response = recommend_side_hustle(payload)
        return response.model_dump()
    except Exception as exc:
        return {"error": str(exc)}


with gr.Blocks(title="FINARY AI") as demo:
    gr.Markdown(
        "# FINARY AI\n"
        "Run classification, insight prediction, and side-hustle recommendations."
    )

    with gr.Tab("Classification"):
        gr.Markdown("Monthly financial condition classification.")
        with gr.Row():
            monthly_income = gr.Number(label="Monthly Income (IDR)")
            monthly_expense_total = gr.Number(label="Monthly Expense Total (IDR)")
        with gr.Row():
            actual_savings = gr.Number(label="Actual Savings (IDR)")
            budget_goal = gr.Number(label="Budget Goal (IDR)")
            emergency_fund = gr.Number(label="Emergency Fund (IDR)")
        classify_btn = gr.Button("Run Classification")
        classify_out = gr.JSON(label="Result")
        classify_btn.click(
            run_classification,
            inputs=[
                monthly_income,
                monthly_expense_total,
                actual_savings,
                budget_goal,
                emergency_fund,
            ],
            outputs=classify_out,
        )

    with gr.Tab("Insight"):
        gr.Markdown("Predict next balance and warning probability.")
        with gr.Row():
            income = gr.Number(label="Income (IDR)")
            expense = gr.Number(label="Expense (IDR)")
        with gr.Row():
            savings = gr.Number(label="Savings (IDR)")
            target_tabungan = gr.Number(label="Target Savings (IDR)")
        with gr.Row():
            loan_payment = gr.Number(label="Loan Payment (IDR)")
            emergency_fund2 = gr.Number(label="Emergency Fund (IDR)")
        insight_btn = gr.Button("Run Insight")
        insight_out = gr.JSON(label="Result")
        insight_btn.click(
            run_insight,
            inputs=[
                income,
                expense,
                savings,
                target_tabungan,
                loan_payment,
                emergency_fund2,
            ],
            outputs=insight_out,
        )

    with gr.Tab("Side Hustle"):
        gr.Markdown("Recommend side-hustle opportunities.")
        experience_level = gr.Dropdown(
            ["Beginner", "Intermediate", "Expert"],
            value="Intermediate",
            label="Experience Level",
        )
        available_hours = gr.Slider(
            minimum=1,
            maximum=60,
            value=10,
            step=1,
            label="Available Hours per Week",
        )
        interest_category = gr.Textbox(label="Interest Category")
        side_hustle_btn = gr.Button("Run Side Hustle")
        side_hustle_out = gr.JSON(label="Result")
        side_hustle_btn.click(
            run_side_hustle,
            inputs=[experience_level, available_hours, interest_category],
            outputs=side_hustle_out,
        )


demo.launch()
