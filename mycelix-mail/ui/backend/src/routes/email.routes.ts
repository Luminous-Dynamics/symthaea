import { Router } from 'express';
import Joi from 'joi';
import { emailController } from '../controllers/email.controller';
import { authenticate } from '../middleware/auth';
import { validate } from '../middleware/validate';

const router = Router();

const sendEmailSchema = Joi.object({
  accountId: Joi.string().required(),
  to: Joi.array().items(Joi.string().email()).required(),
  subject: Joi.string().required(),
  body: Joi.string().required(),
  cc: Joi.array().items(Joi.string().email()).optional(),
  bcc: Joi.array().items(Joi.string().email()).optional(),
  attachments: Joi.array().optional(),
});

const syncSchema = Joi.object({
  accountId: Joi.string().required(),
  folderPath: Joi.string().default('INBOX'),
});

router.use(authenticate);

router.get('/', emailController.getAll);
router.post('/', validate(sendEmailSchema), emailController.send);
router.post('/sync', validate(syncSchema), emailController.sync);
router.get('/:id', emailController.getOne);
router.delete('/:id', emailController.delete);
router.put('/:id/read', emailController.markRead);
router.put('/:id/star', emailController.markStarred);

export { router as emailRoutes };
