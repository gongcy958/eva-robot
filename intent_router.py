from src.eva_robot.domain.intents import Intent, IntentRouter


_router = IntentRouter()


def route_intent(text: str) -> Intent:
    return _router.route(text)
