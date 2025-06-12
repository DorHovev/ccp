from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi import Request
from .metrics import DATA_VALIDATION_FAILURES

async def validation_exception_handler(request: Request, exc: RequestValidationError):
    DATA_VALIDATION_FAILURES.labels(validation_type="pydantic").inc()
    errors = exc.errors()
    for err in errors:
        if 'msg' in err and not isinstance(err['msg'], str):
            err['msg'] = str(err['msg'])
        if 'ctx' in err and err['ctx']:
            for k, v in err['ctx'].items():
                if not isinstance(v, str):
                    err['ctx'][k] = str(v)
    return JSONResponse(
        status_code=422,
        content={"detail": errors, "body": exc.body},
    ) 