import { Router } from 'express';
import Joi from 'joi';
import { authenticate } from '../middleware/auth';
import { validate } from '../middleware/validate';
import { trustController } from '../controllers/trust.controller';

const router = Router();

const summarySchema = Joi.object({
  sender: Joi.string().email().required(),
});

router.use(authenticate);

router.get('/summary', validate(summarySchema, 'query'), trustController.getSummary);
router.post('/cache/clear', trustController.clearCache);
router.get('/health', trustController.health);

export { router as trustRoutes };
