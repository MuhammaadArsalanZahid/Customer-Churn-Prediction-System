# Manager-facing model insights

The features below have the largest mean absolute SHAP values in a validation sample. They describe model associations, not causes.

1. Review customer journeys associated with **categorical__Contract_Month-to-month**. Validate any retention action with a controlled pilot before changing business policy.
2. Review customer journeys associated with **numeric__tenure**. Validate any retention action with a controlled pilot before changing business policy.
3. Review customer journeys associated with **categorical__InternetService_Fiber optic**. Validate any retention action with a controlled pilot before changing business policy.
4. Review customer journeys associated with **categorical__OnlineSecurity_No**. Validate any retention action with a controlled pilot before changing business policy.
5. Review customer journeys associated with **categorical__Contract_Two year**. Validate any retention action with a controlled pilot before changing business policy.

SHAP values do not prove that a feature causes churn. Review fairness, customer impact and retention outcomes before deployment.